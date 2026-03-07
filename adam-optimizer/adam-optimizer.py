import numpy as np

def adam_step(param:np.ndarray, grad:np.ndarray, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    #convert parameters to arrays
    param = np.asarray(param)
    grad = np.asarray(grad)
    m = np.asarray(m)
    v = np.asarray(v)
   
    #first moment and second moment updates
    m_new=beta1*m + (1-beta1)*grad
    v_new=beta2*v + (1-beta2)*grad**2
    mb=m_new/(1-beta1**t)#after bias correction
    vb=v_new/(1-beta2**t)#after bias correction


    param_new=param-(lr*mb)/(np.sqrt(vb)+eps)

    return param_new,m_new,v_new
        

        
    # Write code here
    pass