import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List
from scipy.stats import norm
import numpy as np

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

    def margrabe_formula(self, S1, S2, tau, r, sigma):
        """
        Margrabe formula for exchange option price on two assets, assuming 
        - Geometric Brownian motion
        - Drift of assets unders risk-neutral measure is r
        - BOth assets have the same sigma
        
        Parameters
        ----------
        S1: float
        S2: float
        tau: float
            Time to maturity
        r: float
            risk free rate
        sigma: float
            risk free rate
        """
        N = norm()
        sigma_ = np.sqrt(2*sigma**2)
        d1 = (sigma_**2 / 2 * tau) / (sigma_ * np.sqrt(tau))
        d2 = -d1
        price = S1 * N.cdf(d1) - S2 * N.cdf(d2)
        return price






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
