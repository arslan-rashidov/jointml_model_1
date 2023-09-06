import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
from torch import Tensor
import math


def new_gelu(x):
    # Gaussian Error Linear Units (GELU)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    # LayerNorm but with an optional bias.

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, n_embd, bias, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, bias, dropout):
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias)
        self.attn = SelfAttention(n_embd, n_head, bias, dropout)
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, bias, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


def soft_onehot(x, thresh, temperature):
    return F.softmax(-torch.abs(x - thresh) * temperature, dim=1).to(torch.float32)


class SingleFeatureEncoder(nn.Module):
    def __init__(self, n_embd, n_bins, temperature):
        super().__init__()

        self.encoder = nn.Linear(n_bins, n_embd)
        self.temperature = temperature
        # setting initial distribution to be uniform on -2, 2
        # self.thresholds = nn.Parameter(torch.linspace(-2, 2, n_bins, dtype=torch.float))
        self.thresholds = nn.Parameter(torch.normal(0, 1, size=(n_bins,), dtype=torch.float))

    def forward(self, x):
        bins = soft_onehot(x, self.thresholds, self.temperature)
        result = self.encoder(bins)

        return result


class MultiFeatureEncoder(nn.Module):
    def __init__(self, n_embd, n_bins, n_features, temperature):
        super().__init__()

        self.encoders = nn.ModuleList([SingleFeatureEncoder(n_embd, n_bins, temperature) for _ in range(n_features)])
        self.thresholds = torch.stack([val.thresholds for val in self.encoders])
        n_head = 8  # n_embd // 4
        self.attention = TransformerBlock(n_embd=n_embd, n_head=n_head, bias=False, dropout=0.4)

        self.pos_encoding = torch.zeros(n_features, n_embd)
        pos = torch.arange(0, n_features, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        self.pos_encoding[:, 0::2] = torch.sin(pos * div)
        self.pos_encoding[:, 1::2] = torch.cos(pos * div)
        self.pos_encoding = nn.Parameter(self.pos_encoding)
        self.pos_encoding.requires_grad = False

    def forward(self, x):
        result = torch.stack([encoder(feature.view(-1, 1)) for encoder, feature in zip(self.encoders, x.T)], dim=1)
        result += self.pos_encoding
        result = self.attention(result)

        return result

    def get_thresholds(self):
        return torch.stack([val.thresholds for val in self.encoders])


class EmbeddingModel(nn.Module):
    def __init__(self, n_cat, n_num, n_embd, bias, number_of_bins, temperature):
        super().__init__()

        d_bias = n_num + n_cat
        if n_cat > 0:
            self.category_embeddings = nn.Embedding(n_cat, n_embd)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # take [CLS] token into account
        self.n_num = n_num

        self.n_embd = n_embd
        self.num_encoder = MultiFeatureEncoder(n_embd, number_of_bins, n_num + 1, temperature)
        self.bias = nn.Parameter(Tensor(d_bias, n_embd)) if bias else None

        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
        x_num = torch.cat(
            [torch.ones(len(x_some), 1, device=x_some.device)] + ([] if x_num is None else [x_num]),  # [CLS]
            dim=1,
        )

        x = self.num_encoder(x_num)

        if x_cat is not None:
            x = torch.cat(
                [x, self.category_embeddings(x_cat + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]
        return x


class SoftonehotModel(nn.Module):
    def __init__(
        self, n_cat, n_num, n_embd, n_head, n_layer, bias, dropout, number_of_bins, temperature, output_size, task
    ):
        super().__init__()
        self.embedding_model = EmbeddingModel(n_cat, n_num, n_embd, bias, number_of_bins, temperature)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, bias, dropout) for _ in range(n_layer)])
        self.last_layer = nn.Linear(n_embd * (n_num + 1 + n_cat), output_size, bias=True)
        torch.nn.init.xavier_uniform_(self.last_layer.weight)
        self.last_normalization = LayerNorm(n_embd, bias=bias)
        self.task = task
        self.output_size = output_size

        self.pos_encoding = torch.zeros(n_num + n_cat + 1, n_embd)
        pos = torch.arange(0, n_num + n_cat + 1, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, n_embd, 2).float() * (-math.log(10000.0) / n_embd))
        self.pos_encoding[:, 0::2] = torch.sin(pos * div)
        self.pos_encoding[:, 1::2] = torch.cos(pos * div)
        self.pos_encoding = nn.Parameter(self.pos_encoding)
        self.pos_encoding.requires_grad = False

    def forward(self, x_num, x_cat, first_batch=False):
        x = self.embedding_model(x_num, x_cat)
        x += self.pos_encoding

        for block in self.blocks:
            x = block(x)

        x = self.last_normalization(x)
        x = x.view(x.shape[0], -1)

        if first_batch and self.task == "regression":
            with torch.no_grad():
                x_tmp = self.last_layer(x)
                self.last_layer.bias = nn.Parameter(-torch.mean(x_tmp), requires_grad=True)
        x = self.last_layer(x)
        x = x.squeeze(-1)

        return x

    def set_weights(self, weights):  # TODO: Specify type of weights and return type
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

    def get_weights(self):  # TODO: Specify return type
        return [val.cpu().numpy() for _, val in self.state_dict().items()]


def load_hyperparams(hparams):
    result = {
        "n_embd": int(hparams[0]),
        "dropout": float(hparams[1]),
        "number_of_bins": int(hparams[2]),
        "temperature": float(hparams[3]),
        "weight_decay": 0.1 ** float(hparams[4]),
        "lr": 0.1 ** float(hparams[5]),
        "optimizer": hparams[6],
        "batch_size": int(hparams[7]),
        "n_layer": int(hparams[8]),
    }
    return result


class Metric:
    def __init__(self, metric_f):
        self.num_iter = 0
        self.current_value = 0
        self.metric_f = metric_f

    def update_state(self, target, output):
        self.num_iter += 1
        new_value = self.metric_f(output, target)

        self.current_value = self.current_value * (self.num_iter - 1) / self.num_iter + new_value / self.num_iter

    def get_value(self):
        if getattr(self.current_value, "item", None) is not None:
            return self.current_value.item()
        return self.current_value


# import pandas as pd
# from random import shuffle
# seed = 0
# import numpy as np
# import sklearn.preprocessing
# from sklearn.impute import SimpleImputer
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.nn.init as nn_init
# from torch import Tensor
# import torch.optim as optim
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# import sklearn.metrics as skm
# import math
#
#
# df = pd.read_csv('housing.csv')
# df.fillna(df['total_bedrooms'].mode().values[0], inplace=True)
# part = ['train'] * 13209 + ['val'] * 3303 + ['test'] * 4128
# shuffle(part)
# df['part'] = part
# del df['ocean_proximity']
#
# y_train = df.query('part == "train"')['median_house_value'].values
# y_test = df.query('part == "test"')['median_house_value'].values
# y_val = df.query('part == "val"')['median_house_value'].values
#
# del df['median_house_value']
#
# x_train = df.query('part == "train"').values
# x_test = df.query('part == "test"').values
# x_val = df.query('part == "val"').values
#
# x_train = x_train[:, :-1].astype(float)
# x_test = x_test[:, :-1].astype(float)
# x_val = x_val[:, :-1].astype(float)
#
# normalizer = sklearn.preprocessing.QuantileTransformer(
#     output_distribution='normal',
#     n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
#     subsample=1e9,
#     random_state=seed,
# )
#
# # adding noise
# noise = 0.001
# stds = np.std(x_train, axis=0, keepdims=True)
# noise_std = noise / np.maximum(stds, noise)  # type: ignore[code]
# x_train += noise_std * np.random.default_rng(seed).standard_normal(  # type: ignore[code]
#     x_train.shape
# )
#
# x_train = normalizer.fit_transform(x_train)
# x_val = normalizer.transform(x_val)
# x_test = normalizer.transform(x_test)
#
# mean, std = y_train.mean(), y_train.std()
#
# y_train = (y_train - mean) / std
# y_val = (y_val - mean) / std
# y_test = (y_test - mean) / std
#
# x_train = torch.as_tensor(x_train, dtype=torch.float32)
# x_val = torch.as_tensor(x_val, dtype=torch.float32)
# x_test = torch.as_tensor(x_test, dtype=torch.float32)
#
# y_train = torch.as_tensor(y_train, dtype=torch.float32)
# y_val = torch.as_tensor(y_val, dtype=torch.float32)
# y_test = torch.as_tensor(y_test, dtype=torch.float32)
#
# batch_size = 128
#
# dataset_train = TensorDataset(x_train, y_train)
# loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
#
# dataset_val = TensorDataset(x_val, y_val)
# loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)
#
# dataset_test = TensorDataset(x_test, y_test)
# loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
#
# model = SoftonehotModel(
#      n_cat=0,
#      n_num=8,
#      n_embd=256,
#      n_head=8,
#      n_layer=6,
#      bias=True,
#      dropout=0.3,
#      number_of_bins=256,
#      temperature=3,
#      output_size=1,
#      task='regression'
# )
# optimizer = optim.AdamW(
#     model.parameters(),
#     lr=0.0001,
#     weight_decay=0.000001
# )
#
# def calc_score(model, x, y):
#     res = model(x, None)
#     rmse = skm.mean_squared_error(y, res.detach().numpy()) ** 0.5
#     return rmse * std
#
#
# num_epochs = 1000000
# estimate_samples = 50
#
# train_losses = []
# val_losses = []
#
# train_metrics = []
# val_metrics = []
#
# best_val_metric = 10000
# best_weights = None
# patience = 50
#
# loss_fn = F.mse_loss
#
#
# for epoch in range(num_epochs):
#     print(epoch)
#     model.train()
#     for data in loader_train:
#         optimizer.zero_grad()
#         output = model(data[0], None)
#         loss = loss_fn(output, data[1])
#         loss.backward()
#         optimizer.step()
#
#     with torch.no_grad():
#
#         train_loss = 0
#         val_loss = 0
#
#         for ind, data in enumerate(loader_train):
#             if ind > estimate_samples:
#                 break
#             train_loss += loss_fn(model(data[0], None), data[1])
#
#         for ind, data in enumerate(loader_val):
#             if ind > estimate_samples:
#                 break
#             val_loss += loss_fn(model(data[0], None), data[1])
#
#         train_losses.append((train_loss / estimate_samples).detach().item())
#         val_losses.append((val_loss / estimate_samples).detach().item())
#
#         model.eval()
#
#         train_metric = 0
#         val_metric = 0
#
#         train_metrics.append(calc_score(model, x_train, y_train))
#         val_metric = calc_score(model, x_val, y_val)
#         val_metrics.append(val_metric)
#
#         if val_metric < best_val_metric:
#             best_val_metric = val_metric
#             best_weights = model.state_dict()
#             last_improvement = epoch
#             print(f'New best metric: {val_metric}')
#         else:
#             print(f'last improvement: {last_improvement}')
#
#         if epoch - last_improvement > patience:
#             # Load the best model weights and stop training
#             model.load_state_dict(best_weights)
#             break
#
