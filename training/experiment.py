import torch
import torch.nn as nn
from utils.sparsity import compute_sparsity_loss


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    lambda_sparse,
    device,
    epoch   
):
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_sp_loss = 0.0
    correct = 0
    total_samples = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        ce_loss = criterion(logits, labels)
        sp_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_sparse * sp_loss

        loss.backward()

        # Gradient clipping (from original code)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        total_sp_loss += sp_loss.item()

        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total_samples += labels.size(0)

    n_batches = len(train_loader)

    return {
        'avg_total_loss': total_loss / n_batches,
        'avg_ce_loss': total_ce_loss / n_batches,
        'avg_sp_loss': total_sp_loss / n_batches,
        'train_accuracy': 100.0 * correct / total_samples
    }


def compute_sparsity_info(model, threshold=0.01):
    gates = model.get_all_gates().detach().cpu()
    total_weights = gates.numel()
    pruned_weights = (gates < threshold).sum().item()
    active_weights = total_weights - pruned_weights
    sparsity = 100.0 * pruned_weights / total_weights

    return {
        'total_weights': total_weights,
        'pruned_weights': pruned_weights,
        'active_weights': active_weights,
        'sparsity': sparsity
    }


def run_experiment(lambda_sparse, train_loader, test_loader, device, num_epochs):
    from training.evaluate import evaluate
    from models.network import SelfPruningNetwork

    model = SelfPruningNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    history = {
        'train_acc': [],
        'test_acc': [],
        'sparsity': []
    }

    best_test_acc = 0.0
    best_sparsity_info = None

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            lambda_sparse,
            device,
            epoch
        )

        test_acc = evaluate(model, test_loader, device)
        sparsity_info = compute_sparsity_info(model)

        history['train_acc'].append(train_metrics['train_accuracy'])
        history['test_acc'].append(test_acc)
        history['sparsity'].append(sparsity_info['sparsity'])

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_sparsity_info = sparsity_info

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Acc: {train_metrics['train_accuracy']:.2f}% | "
            f"Test Acc: {test_acc:.2f}% | "
            f"Sparsity: {sparsity_info['sparsity']:.2f}%"
        )

    final_gates = model.get_all_gates().detach().cpu().numpy()
    final_sparsity_info = best_sparsity_info or sparsity_info

    return {
        'lambda': lambda_sparse,
        'history': history,
        'final_test_acc': best_test_acc,
        'final_sparsity': final_sparsity_info['sparsity'],
        'final_gates': final_gates,
        'sparsity_info': final_sparsity_info
    }