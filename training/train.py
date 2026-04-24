import torch
import torch.nn as nn
from utils.sparsity import compute_sparsity_loss

def train_one_epoch(model, loader, optimizer, device, lambda_sparse):
    model.train()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)
        ce_loss = criterion(output, y)
        sp_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_sparse * sp_loss
        loss.backward()

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)