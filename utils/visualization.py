import matplotlib
matplotlib.use('Agg')  # For saving plots without GUI

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PLOT 1: Gate Distribution
# =============================================================================
def plot_gate_distribution(results: list, save_path: str = 'gate_distribution.png'):
    """
    Plots distribution of gate values for all lambda experiments.
    """

    fig, axes = plt.subplots(1, len(results), figsize=(6 * len(results), 5))
    fig.patch.set_facecolor('#0f0f1a')

    if len(results) == 1:
        axes = [axes]

    colors = ['#00d4ff', '#ff6b35', '#7bed9f', '#ffa502']

    for ax, result, color in zip(axes, results, colors):
        ax.set_facecolor('#1a1a2e')

        gates = result['final_gates']

        ax.hist(
            gates,
            bins=100,
            color=color,
            alpha=0.85,
            edgecolor='none',
            density=True
        )

        # Threshold line
        ax.axvline(
            x=0.01,
            color='red',
            linestyle='--',
            linewidth=1.5,
            label='Prune threshold (0.01)'
        )

        pruned_pct = result['final_sparsity']
        active_pct = 100.0 - pruned_pct

        ax.set_title(
            f"λ = {result['lambda']}\n"
            f"Sparsity: {pruned_pct:.1f}%  |  Test Acc: {result['final_test_acc']:.1f}%",
            color='white',
            fontsize=12,
            fontweight='bold',
            pad=10
        )

        ax.set_xlabel('Gate Value', color='#aaaaaa')
        ax.set_ylabel('Density', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')

        ax.legend(
            fontsize=8,
            facecolor='#1a1a2e',
            labelcolor='white'
        )

        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

        # Text box
        textstr = f'Pruned: {pruned_pct:.1f}%\nActive: {active_pct:.1f}%'
        ax.text(
            0.65, 0.85, textstr,
            transform=ax.transAxes,
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round', facecolor='#333355', alpha=0.7)
        )

    plt.suptitle(
        'Gate Value Distribution — Self-Pruning Network\n'
        '(Spike at 0 = pruned weights, cluster near 1 = active weights)',
        color='white',
        fontsize=13,
        fontweight='bold',
        y=1.02
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='#0f0f1a',
        edgecolor='none'
    )
    plt.close()

    print(f"→ Gate distribution plot saved: {save_path}")


# =============================================================================
# PLOT 2: Training Curves
# =============================================================================
def plot_training_curves(results: list, save_path: str = 'training_curves.png'):
    """
    Plots training accuracy, test accuracy, and sparsity curves.
    """

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('#0f0f1a')

    colors = ['#00d4ff', '#ff6b35', '#7bed9f', '#ffa502']
    titles = ['Training Accuracy (%)', 'Test Accuracy (%)', 'Sparsity Level (%)']
    keys = ['train_acc', 'test_acc', 'sparsity']

    for ax, title, key in zip(axes, titles, keys):
        ax.set_facecolor('#1a1a2e')

        for result, color in zip(results, colors):
            epochs = range(1, len(result['history'][key]) + 1)

            ax.plot(
                epochs,
                result['history'][key],
                color=color,
                linewidth=2,
                label=f"λ={result['lambda']}"
            )

        ax.set_title(title, color='white', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', color='#aaaaaa')
        ax.tick_params(colors='#aaaaaa')

        ax.legend(
            facecolor='#1a1a2e',
            labelcolor='white',
            fontsize=9
        )

        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    plt.suptitle(
        'Training Dynamics — Self-Pruning Network (λ Comparison)',
        color='white',
        fontsize=14,
        fontweight='bold'
    )

    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches='tight',
        facecolor='#0f0f1a',
        edgecolor='none'
    )
    plt.close()

    print(f"→ Training curves plot saved: {save_path}")


# =============================================================================
# TABLE PRINT
# =============================================================================
def print_results_table(results: list):
    """
    Prints summary table of experiments.
    """

    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY TABLE")
    print("=" * 70)

    print(f"{'Lambda':<12} {'Test Accuracy':>15} {'Sparsity':>12} {'Active Weights':>20}")
    print("-" * 70)

    for r in results:
        total = r['sparsity_info']['total_weights']
        active = r['sparsity_info']['active_weights']

        print(
            f"{r['lambda']:<12} "
            f"{r['final_test_acc']:>13.2f}% "
            f"{r['final_sparsity']:>10.2f}% "
            f"{active:>10,} / {total:,}"
        )

    print("=" * 70)