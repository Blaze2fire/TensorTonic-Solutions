import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    max=torch.max(logits,dim=-1,keepdim=True).values#calculate maximum without changing dimensions
    shift=torch.exp(logits-max)
    smax=shift/shift.sum(dim=1,keepdim=True)
    return smax
    pass
