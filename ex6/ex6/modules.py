import torch
import torch.nn as nn


class Dropout(nn.Module):
    
    def __init__(self, p=0.1):
        super().__init__()
        # store p
        self.p = p
        
    def forward(self, x):
        # In training mode
        if self.training:
        
        #, set each value 
        # independently to 0 with probability p
        # and scale the remaining values 
        # according to the lecture
        
            rand = torch.rand_like(x)
            dropout = torch.ones_like(x)
             
            dropout[rand <= self.p] = 0
            dropout /= 1-self.p

            return x * dropout
        
        # In evaluation mode, return the
        # unmodified input
        else:
            return x
    