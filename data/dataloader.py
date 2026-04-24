import torchvision
import torchvision.transforms as transforms
import torch

def get_cifar10_loaders(batch_size=128):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)

    test = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size)

    return train_loader, test_loader