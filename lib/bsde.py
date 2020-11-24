import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod

from lib.networks import FFN
from lib.functions import BaseFinal


class FBSDE(nn.Module):

    def __init__(self, d: int, ffn_hidden: List[int]):
        super().__init__()
        self.d = d

        self.f = FFN(sizes = [d+1]+ffn_hidden+[1]) # +1 is for time
        self.dfdx = FFN(sizes = [d+1]+ffn_hidden+[2])

    
    @abstractmethod
    def drift(self, x):
        """
        Code here the drift of the SDE associated to the linear PDE
        """
        ...

    @abstractmethod
    def diffusion(self, x):
        """
        Code here the diffusion of the SDE associated to the linear PDE
        """
        ...


    def sdeint(self, ts, x0):
        """
        Euler scheme to solve the SDE.
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        brownian: Optional. 
            torch.tensor of shape (batch_size, N, d)
        Note
        ----
        I am assuming uncorrelated Brownian motion
        """
        x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new = x[:,-1,:] + self.drift(x[:,-1,:])*h + self.diffusion(x[:,-1,:])*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments
    
    def bsdeint(self, ts: torch.Tensor, x0: torch.Tensor, final: BaseFinal): 
        """
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        option: object of class option to calculate payoff
        
        """

        x, brownian_increments = self.sdeint(ts, x0)
        final_value = final(x[:,-1,:]) # (batch_size, 1)
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        
        Y = self.f(tx) # (batch_size, L, 1)
        Z = self.dfdx(tx) # (batch_size, L, dim)

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts):
            if t==ts[-1]:
                target = final_value
            else:
                target = Y[:,idx+1,:].detach()
            stoch_int = torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
            pred = Y[:,idx,:] + stoch_int # if t==ts[-1], then it is already taken into account that stoch_int=0, because the increment of Brownian motion is 0, therefore we are indeed comparing against the payoff
            loss += loss_fn(pred, target)
        return loss, Y, final_value
            
            
    def conditional_expectation(self, ts: torch.Tensor, x0: torch.Tensor, final: BaseFinal): 
        """
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (batch_size, d)
        option: object of class option to calculate payoff
        
        """

        x, brownian_increments = self.sdeint(ts, x0)
        final_value = final(x[:,-1,:]) # (batch_size, 1)
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        Y = self.f(tx) # (batch_size, L, 1)

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts):
            target = final_value 
            pred = Y[:,idx,:] 
            loss += loss_fn(pred, target)
        return loss, Y, final_value




class FBSDE_Brownian(FBSDE):

    def __init__(self, d: int, ffn_hidden: List[int]):
        """
        FBSDE of the Brownian motion
        dXt = dWt
        """
        super(FBSDE_Brownian, self).__init__(d=d, ffn_hidden=ffn_hidden)
    
    
    def drift(self, x):
        """
        """
        return 0

    def diffusion(self, x):
        """
        """
        return torch.ones_like(x)
    
