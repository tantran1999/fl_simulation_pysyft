from syft.core.plan.plan_builder import ROOT_CLIENT
import torch as th

# Set updated params to model
def set_params(model, params):
    for p, p_new in zip(model.parameters(), params):
        p.data = p_new.data

# Loss function

def cross_entropy_loss(logits, targets, batch_size=100):
    norm_logits = logits - logits.max()
    log_probs = norm_logits - norm_logits.exp().sum(dim=1, keepdim=True).log()
    return -(targets * log_probs).sum() / batch_size

# Optimizer - Update gradients
def sgd_step(model, lr=0.05):
    with ROOT_CLIENT.torch.no_grad():
        for p in model.parameters():
            p.data = p.data - lr * p.grad
            p.grad = th.zeros_like(p.grad.get())
