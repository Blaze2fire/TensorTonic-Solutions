import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x=np.asarray(x,dtype=float)
    x_sig=1/(1+np.exp(-x))

    return x_sig
    # Write code here
    pass