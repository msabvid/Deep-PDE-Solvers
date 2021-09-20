import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

@dataclass
class BaseOption:
    pass
    
    @abstractmethod
    def payoff(self, x: torch.Tensor, **kwargs):
        ...



class Lookback(BaseOption):
    
    def __init__(self, idx_traded: List[int]=None):
        self.idx_traded = idx_traded # indices of traded assets. If None, then all assets are traded
    
    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Path history. Tensor of shape (batch_size, N, d) where N is path length
        Returns
        -------
        payoff: torch.Tensor
            lookback option payoff. Tensor of shape (batch_size,1)
        """
        if self.idx_traded:
            basket = torch.sum(x[..., self.idx_traded],2) # (batch_size, N)
        else:
            basket = torch.sum(x,2) # (batch_size, N)
        payoff = torch.max(basket, 1)[0]-basket[:,-1] # (batch_size)
        return payoff.unsqueeze(1) # (batch_size, 1)


class Exchange(BaseOption):
    
    def __init__(self):
        pass

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, d) 
        Returns
        -------
        payoff: torch.Tensor
            exchange option payoff. Tensor of shape (batch_size,1)
        """
        #assert x.shape[1]==2, "need dim=2"
        payoff = torch.clamp(x[:,0]-x[:,1:].mean(1), 0)
        return payoff.unsqueeze(1) # (batch_size, 1)


class Basket(BaseOption):
    
    def __init__(self, K):
        self.K = K

    def payoff(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Asset price at terminal time. Tensor of shape (batch_size, d) 
        Returns
        -------
        payoff: torch.Tensor
            basket option payoff. Tensor of shape (batch_size,1)
        """
        payoff = torch.sum(x, 1, keepdim=True)
        return torch.clamp(payoff-self.K, 0) # (batch_size, 1)
