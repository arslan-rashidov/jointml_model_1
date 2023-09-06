import torch
import torch.nn.functional as F
from torch import optim

from softonehot_model import SoftonehotModel

init_parameters = {
    'n_cat': 0,
    'n_num': 8,
    "n_embd": 64,
    "n_head": 8,
    "dropout": 0.2,
    "bias": True,
    "number_of_bins": 16,
    "temperature": 16,
    "n_layer": 4,
    "output_size": 2,
    "task": "binary_classification"
}

train_parameters = {
    "weight_decay": 3,
    "lr": 0.0001,
    "optimizer": 'Adam',
    "batch_size": 4,
    "num_epochs": 16,
    'patience': True
}


class Client:
    def __init__(self):
        self.model = SoftonehotModel(
            n_cat=init_parameters['n_cat'],
            n_num=init_parameters['n_num'],
            n_embd=init_parameters['n_embd'],
            n_head=init_parameters['n_head'],
            n_layer=init_parameters['n_layer'],
            bias=init_parameters['bias'],
            dropout=init_parameters['dropout'],
            number_of_bins=init_parameters['number_of_bins'],
            temperature=init_parameters['temperature'],
            output_size=init_parameters['output_size'],
            task=init_parameters['task'],
        )

        if init_parameters['optimizer'] == "Adam":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=train_parameters["lr"],
                weight_decay=train_parameters['weight_decay'],
            )

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def train(self, train_ins):
        self.model.train()
        loss_f = F.mse_loss()
        patience = train_parameters['patience']
        num_epochs = train_parameters['num_epochs']

        if patience is not None:
            best_loss = 10000
            best_weights = None
            last_improvement = 0
        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data, None)
                loss = loss_f(output, target)
                loss.backward()
                self.optimizer.step()

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for (
                    data,
                    target,
            ) in self.test_loader:
                output = self.model(data, None)
                for metric in self.metrics.values():
                    metric.update_state(target, output)

        return self.metrics["loss"].get_value()

    def get_prediction(self):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for (
                    data,
                    target,
            ) in self.test_loader:
                output = self.model(data, None)
                predictions.append(list(output))
                for metric in self.metrics.values():
                    metric.update_state(target, output)

        return predictions
