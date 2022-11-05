import math
import os
from pathlib import Path

from torch.utils.data import DataLoader
import torch.nn as nn
import torch

model_path = Path(__file__).parent / 'models'
model_path.mkdir(exist_ok=True, parents=True)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def mse(loader: DataLoader, model: nn.Module) -> float:
    device = get_device()
    loss_fn = nn.MSELoss(reduction='sum')
    loss_sum = 0
    num_items = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).unsqueeze(1)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_sum += loss.item()
        num_items += x.shape[0]
    result = loss_sum / num_items
    return result

def save_model(model: nn.Module, name: str):
    torch.save(model.state_dict(), model_path / f'{name}.pt')

def load_model(model: nn.Module, name: str) -> nn.Module:
    state_dict = torch.load(model_path / f'{name}.pt')
    model.load_state_dict(state_dict)
    return model

def train(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int = 2000,
        epochs_until_eval: int = 5,
        name: str = 'temporary'
):
    device = get_device()
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss().to(device)
    best_epoch = -1
    best_mse_val = math.inf

    for i in range(num_epochs):
        # training
        for x, y in train_loader:
            optim.zero_grad()
            x, y = x.to(device), y.to(device).unsqueeze(1)
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()

        if i % epochs_until_eval == 0:
            mse_val = mse(val_loader, model)
            print(f'Epoch {i}: {mse_val} mse val loss')
            if mse_val < best_mse_val:  # save best model
                save_model(model, f'{name}_{i}')
                old_best = model_path / f'{name}_{best_epoch}.pt'
                if old_best.exists():
                    os.remove(old_best)
                best_mse_val = mse_val
                best_epoch = i

    print(f'loading best model from epoch {best_epoch}')
    model = load_model(model, f'{name}_{best_epoch}')
    mse_test = mse(test_loader, model)
    print(f'Testing performance: {mse_test}')








