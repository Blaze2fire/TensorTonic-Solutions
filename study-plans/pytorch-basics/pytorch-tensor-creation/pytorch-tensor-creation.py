import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns: list
    """
    if method=='full':
        return torch.full(shape,value).tolist()
    factory=getattr(torch,method,None)
    return factory(shape).tolist()