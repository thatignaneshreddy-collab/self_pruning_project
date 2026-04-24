import torch
import os

from data.dataloader import get_cifar10_loaders
from training.experiment import run_experiment
from utils.visualization import (
    plot_gate_distribution,
    plot_training_curves,
    print_results_table
)


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       SELF-PRUNING NEURAL NETWORK — CIFAR-10                ║")
    print("║       Dynamic Weight Pruning via Learnable Gates            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Device Setup ────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load Data ───────────────────────────────────────────────────
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=256)

    print(f"Training samples: {len(train_loader.dataset):,}")
    print(f"Test samples    : {len(test_loader.dataset):,}")

    # ── Lambda Values ───────────────────────────────────────────────
    lambda_values = [1e-5, 1e-4, 1e-3]

    # Number of epochs (same as original)
    NUM_EPOCHS = 30

    # ── Run Experiments ─────────────────────────────────────────────
    all_results = []

    for lam in lambda_values:
        print(f"\n{'='*60}")
        print(f"Running experiment with λ = {lam}")
        print(f"{'='*60}")

        result = run_experiment(
            lambda_sparse=lam,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            num_epochs=NUM_EPOCHS
        )

        all_results.append(result)

    # ── Print Results Table ─────────────────────────────────────────
    print_results_table(all_results)

    # ── Save Plots ──────────────────────────────────────────────────
    print("\nGenerating plots...")

    os.makedirs('outputs', exist_ok=True)

    plot_gate_distribution(
        all_results,
        save_path='outputs/gate_distribution.png'
    )

    plot_training_curves(
        all_results,
        save_path='outputs/training_curves.png'
    )

    print("\n✓ All experiments complete!")
    print("✓ Plots saved in 'outputs/' folder")


if __name__ == "__main__":
    main()