import torch

def gradient_accumulation(w_init, micro_batches, lr, accum_steps):
    """
    Returns: tuple of (updated_weights_list, last_avg_gradient_list)
    """
    w=torch.tensor(w_init,dtype=torch.float32,requires_grad=True)
    grad_accum = None
    for i,(inputs,targets) in enumerate(micro_batches):
        inputs=torch.tensor(inputs,dtype=torch.float32)
        targets=torch.tensor(targets,dtype=torch.float32)
        
        preds=torch.dot(w,inputs)
        loss=(preds-targets)**2
        loss.backward()

        if (i+1)%accum_steps==0:
            grad_accum=w.grad.clone()/accum_steps
            with torch.no_grad():
                w-=lr*grad_accum
            w.grad.zero_()

    return w.detach().tolist(),grad_accum.tolist()
        
        
        
    
    
    pass
