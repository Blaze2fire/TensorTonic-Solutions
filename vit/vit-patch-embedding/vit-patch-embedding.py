import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    #Perform the split into patches?how?
    #Shape of image is B,H,W,C
    B,H,W,C=image.shape
    nh=H//patch_size
    nw=W//patch_size
    image = image[:, :nh*patch_size, :nw*patch_size, :]
    
    

    patches=image.reshape(B,nh,patch_size,nw,patch_size,C)
    flattened_patches=patches.transpose(0,1,3,2,4,5).reshape(B,nh*nw,patch_size**2*C)
    z=flattened_patches@W_proj+bias
    return z
    
    
    pass