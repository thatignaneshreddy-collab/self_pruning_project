================================================================================
  SELF-PRUNING NEURAL NETWORK — CIFAR-10
  Dynamic Weight Pruning via Learnable Gates
================================================================================

OVERVIEW
--------
This project implements a self-pruning fully-connected neural network for
CIFAR-10 image classification. Each weight connection has a learnable gate
(sigmoid-activated score) that is trained alongside the weights. A sparsity
loss term encourages gates toward zero, effectively pruning unimportant
connections during training.

ARCHITECTURE
------------
  Input:     3072  (32x32x3 CIFAR-10 flattened)
  FC1:       3072 -> 1024  + BatchNorm + ReLU
  FC2:       1024 -> 512   + BatchNorm + ReLU
  FC3:       512  -> 256   + BatchNorm + ReLU
  FC4:       256  -> 128   + BatchNorm + ReLU
  FC5:       128  -> 10    (output logits)

All linear layers use PrunableLinear, which multiplies weights by a sigmoid gate
matrix before the forward pass. The sparsity loss is the sum of all gate values.

PROJECT STRUCTURE
-----------------
  main.py                     Entry point — runs all experiments
  requirements.txt            Python dependencies

  models/
    prunable_linear.py        Custom linear layer with learnable gates
    network.py                SelfPruningNetwork model definition

  training/
    train.py                  Single epoch training loop
    evaluate.py               Test set evaluation
    experiment.py             Full experiment runner (train + evaluate)

  utils/
    sparsity.py               Sparsity loss computation
    visualization.py          Plotting and result table printing

  data/
    dataloader.py             CIFAR-10 train/test DataLoaders

  outputs/                    Generated plots (created automatically)

DEPENDENCIES
------------
  torch
  torchvision
  numpy
  matplotlib

Install with:
  pip install -r requirements.txt

HOW TO RUN
----------
  python main.py

The script will:
  1. Load CIFAR-10 dataset (auto-downloaded on first run)
  2. Run 3 experiments with different sparsity penalties:
       λ = 1e-5, 1e-4, 1e-3
  3. Train each model for 30 epochs
  4. Print a results table
  5. Save plots to outputs/:
       - gate_distribution.png
       - training_curves.png

RUNNING ON CPU
--------------
If CUDA is unavailable, the code automatically falls back to CPU. Training
will be slower but fully functional.

SPARSITY MECHANISM
------------------
  - Each weight has a gate score g in (-inf, +inf)
  - Effective weight = weight * sigmoid(g)
  - Sparsity loss = sum(sigmoid(g)) over all gates
  - The optimizer pushes gate scores negative to prune weights
  - A gate < 0.01 is considered pruned

RESULTS INTERPRETATION
----------------------
  - Higher λ  →  more sparsity  →  fewer active weights  →  potentially lower accuracy
  - Lower λ   →  less sparsity  →  more active weights   →  higher accuracy but less compression

VIRTUAL ENVIRONMENTS
--------------------
The .gitignore excludes common virtual environment folders:
  myenv/, venv312/, venv312_new/, .venv/, venv/, env/

Create and use a virtual environment (recommended):
  python -m venv venv
  venv\Scripts\activate        (Windows)
  source venv/bin/activate     (Linux/Mac)
  pip install -r requirements.txt

================================================================================

