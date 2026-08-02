import torch
import math

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    """
    var=X.var(dim=0,unbiased=False)
    mean=X.mean(dim=0)#calculate mean and var
    X_norm=(X-mean)/torch.sqrt(var+eps)
    return gamma*X_norm+beta
    
    pass
