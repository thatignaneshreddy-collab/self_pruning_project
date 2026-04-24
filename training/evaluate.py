import torch

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            _, pred = output.max(1)

            correct += pred.eq(y).sum().item()
            total += y.size(0)

    return 100 * correct / total