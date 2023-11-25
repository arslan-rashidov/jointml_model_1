import pandas as pd
from random import shuffle
import sklearn
from sklearn import preprocessing
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def get_dataloaders(dataset_path):
    df = pd.read_csv(dataset_path)
    df.fillna(df["total_bedrooms"].mode().values[0], inplace=True)

    part = ["train"] * 13209 + ["val"] * 3303 + ["test"] * 4128
    shuffle(part)
    df["part"] = part
    del df["ocean_proximity"]

    y_train = df.query('part == "train"')["median_house_value"].values
    y_test = df.query('part == "test"')["median_house_value"].values
    y_val = df.query('part == "val"')["median_house_value"].values
    del df["median_house_value"]

    x_train = df.query('part == "train"').values
    x_test = df.query('part == "test"').values
    x_val = df.query('part == "val"').values

    x_train = x_train[:, :-1].astype(float)
    x_test = x_test[:, :-1].astype(float)
    x_val = x_val[:, :-1].astype(float)

    normalizer = preprocessing.QuantileTransformer(
        output_distribution="normal",
        n_quantiles=max(min(x_train.shape[0] // 30, 1000), 10),
        subsample=1e9,
        random_state=42,
    )

    # adding noise
    noise = 0.001
    stds = np.std(x_train, axis=0, keepdims=True)
    noise_std = noise / np.maximum(stds, noise)
    x_train += noise_std * np.random.default_rng(42).standard_normal(x_train.shape)

    x_train = normalizer.fit_transform(x_train)
    x_val = normalizer.transform(x_val)
    x_test = normalizer.transform(x_test)

    mean, std = y_train.mean(), y_train.std()
    y_train = (y_train - mean) / std
    y_val = (y_val - mean) / std
    y_test = (y_test - mean) / std

    x_train = torch.as_tensor(x_train, dtype=torch.float32)
    x_val = torch.as_tensor(x_val, dtype=torch.float32)
    x_test = torch.as_tensor(x_test, dtype=torch.float32)

    y_train = torch.as_tensor(y_train, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32)
    y_test = torch.as_tensor(y_test, dtype=torch.float32)

    dataset_train = TensorDataset(x_train, y_train)

    dataset_val = TensorDataset(x_val, y_val)

    dataset_test = TensorDataset(x_test, y_test)

    return dataset_train, dataset_val, dataset_test
