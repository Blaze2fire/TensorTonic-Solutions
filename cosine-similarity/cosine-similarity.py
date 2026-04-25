import numpy as np

def cosine_similarity(a, b):
    """
    Compute cosine similarity between two 1D NumPy arrays.
    Returns: float in [-1, 1]
    """
    norm_a,norm_b=np.linalg.norm(a),np.linalg.norm(b)
    
    if norm_a==0 or norm_b==0 :
        return 0

    cosine_similarity=np.dot(a,b)/(norm_a*norm_b)

    return np.clip(cosine_similarity,-1,1)

    
    pass