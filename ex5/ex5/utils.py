import torch
import torch.nn as nn
import numpy as np

def init_weights(m):
    if isinstance(m, nn.Linear):
        # set the layer's weights as discussed in the lecture
        weights = m.weight
        n = weights.shape[1]
        mean = 0
        var = torch.as_tensor(2 / n)
        std = np.sqrt(var)
        
        nn.init.normal_(weights, mean, std)


class BatchNorm(nn.Module):
    
    def __init__(self, num_channels):
        super().__init__()
        # set theta_mu and theta_sigma such that the output of
        # forward initially is zero centered and 
        # normalized to variance 1
        theta_mu = torch.zeros(num_channels)
        theta_sigma = torch.ones(num_channels)
        self.theta_mu = nn.Parameter(theta_mu)
        self.theta_sigma = nn.Parameter(theta_sigma)
        self.running_mean = None
        self.running_var = None
        self.eps = 1e-6
        
    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0)

            # specify behavior at training time
            if self.running_mean is None:
                # set the running stats to stats of x
                self.running_mean = mean
                self.running_var = var
            else:
                # update the running stats by setting them
                # to the weighted sum of 0.9 times the
                # current running stats and 0.1 times the
                # stats of x 
                update1 = 0.1
                update2 = 1. - update1
                self.running_mean = update2 * self.running_mean + update1 * mean
                self.running_var = update2 * self.running_var + update1 * var
        else:
            if self.running_mean is None:
                # normalized wrt to stats of
                # current batch x
                pass
            else:
                # use running stats for normalization
                mean = self.running_mean
                var = self.running_var
            
        x = self.theta_sigma * (x - mean) / np.sqrt(var + self.eps) + self.theta_mu
        return x    
            
    