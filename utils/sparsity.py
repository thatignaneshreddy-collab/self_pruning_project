def compute_sparsity_loss(model):
    gates = model.get_all_gates()
    return gates.sum()