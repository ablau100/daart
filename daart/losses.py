"""Custom losses for PyTorch models."""

import numpy as np
import torch
from typeguard import typechecked
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# to ignore imports for sphix-autoapidoc
__all__ = ['kl_div_to_std_normal', 'FocalLoss']


@typechecked
def kl_div_to_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Compute element-wise KL(q(z) || N(0, 1)) where q(z) is a normal parameterized by mu, logvar.

    Parameters
    ----------
    mu : torch.Tensor
        mean parameter of shape (n_sequences, sequence_length, n_dims)
    logvar : torch.Tensor
        log variance parameter of shape (n_sequences, sequence_length, n_dims)

    Returns
    -------
    torch.Tensor
        KL divergence summed across dims, averaged across batch

    """
    kl = 0.5 * torch.sum(logvar.exp() - logvar + mu.pow(2) - 1, dim=-1)
    return torch.mean(kl)



class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True, ignore_index=0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
            

        #input = input[target!=self.ignore_index]
        #target = target[target!=self.ignore_index]
        target = target.view(-1,1)
        
        #print('input', input, input.shape)
        
        logpt = F.log_softmax(input, dim=1)
        
        #print('logpt', logpt, logpt.shape)
        
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()