import torch
from data.dataloader import get_cifar10_loaders
from training.experiment import run_experiment

device = torch.device('cpu')
train_loader, test_loader = get_cifar10_loaders(batch_size=256)

# Quick test with just 1 epoch and 1 lambda
result = run_experiment(
    lambda_sparse=1e-4,
    train_loader=train_loader,
    test_loader=test_loader,
    device=device,
    num_epochs=1
)

print("\n=== Test completed successfully ===")
print(f"Final test accuracy: {result['final_test_acc']:.2f}%")
print(f"Final sparsity: {result['final_sparsity']:.2f}%")

