import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    x,y=np.asarray(x),np.asarray(y)
    return float(np.dot(x,y))
    pass