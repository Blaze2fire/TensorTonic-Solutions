import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p
    
    def forward(self, x):
        """
        Returns: tensor with dropout applied
        """
        if self.training==True:# Check if dropout applicable
            if self.p==1:
               return torch.zeros_like(x)
            else:
                mask=(torch.rand_like(x)>self.p).float() #Create Dropout Mask of T and F
                 #Convert to float
                y=(mask*x)/(1-self.p) #scaled and dropped out tensor
                return y
        else:
            return x
        
        

        pass
