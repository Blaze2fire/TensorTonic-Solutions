import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    m,n=np.shape(A)
    A_t=np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            A_t[i][j]=A[j][i]

    return A_t
            
    pass
