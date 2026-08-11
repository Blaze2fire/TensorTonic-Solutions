import torch
import torch.nn as nn

def train_epoch(model, dataloader, criterion, optimizer):
    """
    Returns: average loss over all batches (float)
    """
    
    #forward pass loop over dataloader
    total_loss=0
    for inputs,targets in dataloader:#runs as many times as no of batches
        optimizer.zero_grad()#Reset gradients from previous batch
        predictions=model(inputs)# pred per batch
        loss=criterion(predictions,targets)
        total_loss+=loss.item()#computes loss 
        loss.backward()# gradient computation
        optimizer.step() #weight update

    return total_loss/len(dataloader)

    
    #loss computation
    pass
