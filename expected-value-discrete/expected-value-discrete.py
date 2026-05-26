import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if not np.isclose(np.sum(p),1) or (np.shape(x)!=np.shape(p)):
        raise ValueError
        
    return np.dot(x,p)
    pass
