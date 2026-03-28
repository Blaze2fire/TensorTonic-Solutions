def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    
    x=float(x0) 
    if not (a>0) and (lr>0) and (steps>=1):
        raise ValueError
        
    for i in range(steps):
        x=x-lr*(2*a*x+b)

    return x
        
    
    pass