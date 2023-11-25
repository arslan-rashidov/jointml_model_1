from __future__ import annotations

from typing import Union

from joint_ml._metric import Metric
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn.functional as F

from softonehot_data import get_dataloaders
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

def load_model(n_cat, n_num, n_embd, n_head, n_layer, bias, dropout, number_of_bins, temperature, output_size, task) -> torch.nn.Module:
    model = SoftonehotModel(
        n_cat, n_num, n_embd, n_head, n_layer, bias, dropout, number_of_bins, temperature, output_size, task
    )
    return model


def get_dataset(dataset_path: str, with_split: bool) -> (Dataset, Dataset, Dataset):
    train_dataset, val_dataset, test_dataset = None, None, None
    if with_split:
        train_dataset, val_dataset, test_dataset = get_dataloaders(dataset_path)

    return train_dataset, val_dataset, test_dataset


def train(model: torch.nn.Module, train_set: torch.utils.data.Dataset, batch_size, num_epochs, patience,
          lr, valid_set: torch.utils.data.Dataset = None) -> tuple[list[Metric], torch.nn.Module]:
    model.train()
    loss_f = F.mse_loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    if patience is not None:
        best_loss = 10000
        best_weights = None
        last_improvement = 0
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = model(data, None)
            loss = loss_f(output, target)
            loss.backward()
            optimizer.step()

    return [], model


def test(model: torch.nn.Module, test_set: torch.utils.data.Dataset) -> Union[
    list[Metric], tuple[list[Metric], list]]:
    model.eval()

    metric = Metric('mse_train')

    test_dataloader = DataLoader(test_set)

    with torch.no_grad():
        for (
                data,
                target,
        ) in test_dataloader:
            output = model(data, None)
            metric.log_value(F.mse_loss(target, output))

    return [metric]


def get_prediction(model: torch.nn.Module, dataset_path: str) -> list:
    model.eval()

    preds = []

    train_set, val_set, test_set = get_dataset(dataset_path)
    test_dataloader = DataLoader(test_set)

    with torch.no_grad():
        for (
                data,
                target,
        ) in test_dataloader:
            output = model(data, None)
            preds.append(output)

    return preds