import torch
import torch.nn as nn
import signatory
from typing import Tuple, Optional, List
from abc import abstractmethod

from lib.networks import FFN
from lib.options import BaseOption



class FBSDE(nn.Module):

    def __init__(self, d: int, mu: float, ffn_hidden: List[int]):
        super().__init__()
        self.d = d
        self.mu = mu # risk free rate
        

        self.f = FFN(sizes = [d+1]+ffn_hidden+[1]) # +1 is for time
        self.dfdx = FFN(sizes = [d+1]+ffn_hidden+[2])

    @abstractmethod
    def sdeint(self, ts, x0):
        """
        Code here the SDE that the underlying assets follow
        """
        ...
    
    def bsdeint(self, ts: torch.Tensor, x0: torch.Tensor, option: BaseOption): 
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
        payoff = option.payoff(x[:,-1,:]) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        
        Y = self.f(tx) # (batch_size, L, 1)
        Z = self.dfdx(tx) # (batch_size, L, dim)

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts):
            if t==ts[-1]:
                target = payoff
            else:
                discount_factor = torch.exp(-self.mu*(ts[idx+1]-t))
                target = discount_factor*Y[:,idx+1,:].detach()
            stoch_int = torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
            pred = Y[:,idx,:] + stoch_int # if t==ts[-1], then it is already taken into account that stoch_int=0, because the increment of Brownian motion is 0, therefore we are indeed comparing against the payoff
            loss += loss_fn(pred, target)
        return loss, Y, payoff
            
            
    def l2_proj(self, ts: torch.Tensor, x0: torch.Tensor, option: BaseOption): 
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
        payoff = option.payoff(x[:,-1,:]) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2)
        Y = self.f(tx) # (batch_size, L, 1)

        loss_fn = nn.MSELoss()
        loss = 0
        for idx,t in enumerate(ts):
            discount_factor = torch.exp(-self.mu*(ts[-1]-t))
            target = discount_factor*payoff 
            pred = Y[:,idx,:] 
            loss += loss_fn(pred, target)
        return loss, Y, payoff

    def unbiased_price(self, ts: torch.Tensor, x0:torch.Tensor, option: BaseOption, MC_samples: int, method: str = 'bsde'):
        """
        We calculate an unbiased estimator of the price at time t=0 (for now) using Monte Carlo, and the stochastic integral as a control variate
        Parameters
        ----------
        ts: troch.Tensor
            timegrid. Vector of length N
        x0: torch.Tensor
            initial value of SDE. Tensor of shape (1, d)
        option: object of class option to calculate payoff
        MC_samples: int
            Monte Carlo samples
        """
        assert x0.shape[0] == 1, "we need just 1 sample"
        x0 = x0.repeat(MC_samples, 1)
        x, brownian_increments = self.sdeint(ts, x0)
        payoff = option.payoff(x[:,-1,:]) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2) # (batch_size, L, dim)
        if method == 'bsde':
            with torch.no_grad():
                Z = self.dfdx(tx) # (batch_size, L, dim)
        elif method == 'l2_proj':
            #tx.requires_grad_(True)
            Z = []
            for j in range(tx.shape[1]):
                input_ = tx[:,j,:]
                input_.requires_grad_(True)
                Y = self.f(input_) # (batch_size, 1)
                Z.append(torch.autograd.grad(Y.sum(), input_, allow_unused=True)[0][:,1:])
            Z = torch.stack(Z, 1)
        else:
            raise ValueError('Unknown method {}'.format(method))
        stoch_int = 0
        for idx,t in enumerate(ts):
            discount_factor = torch.exp(-self.mu*t)
            stoch_int += discount_factor * torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
        
        return torch.exp(-self.mu*ts[-1])*payoff, torch.exp(-self.mu*ts[-1])*payoff-stoch_int # stoch_int has expected value 0, thus it doesn't add any bias to the MC estimator, and it is correlated with payoff




class FBSDE_BlackScholes(FBSDE):

    def __init__(self, d: int, mu: float, sigma: float, ffn_hidden: List[int]):
        super(FBSDE_BlackScholes, self).__init__(d=d, mu=mu, ffn_hidden=ffn_hidden)
        self.sigma = sigma # change it to a torch.parameter to solve a parametric family of PPDEs
    
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
            x_new = x[:,-1,:] + self.mu*x[:,-1,:]*h + self.sigma*x[:,-1,:]*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

    
