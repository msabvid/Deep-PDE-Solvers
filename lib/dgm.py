"""
Deep Galerkin Method: https://arxiv.org/abs/1708.07469
"""

import torch
import torch.nn as nn
import numpy as np
import tqdm

from lib.options import BaseOption

class DGM_Layer(nn.Module):
    
    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()
        
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))
            

        self.gate_Z = self.layer(dim_x+dim_S, dim_S)
        self.gate_G = self.layer(dim_x+dim_S, dim_S)
        self.gate_R = self.layer(dim_x+dim_S, dim_S)
        self.gate_H = self.layer(dim_x+dim_S, dim_S)
            
    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l
    
    def forward(self, x, S):
        x_S = torch.cat([x,S],1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        
        input_gate_H = torch.cat([x, S*R],1)
        H = self.gate_H(input_gate_H)
        
        output = ((1-G))*H + Z*S
        return output


class Net_DGM(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        output = self.output_layer(S4)
        return output

def get_gradient(output, x):
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))    
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian


class PDE_DGM_BlackScholes(nn.Module):

    def __init__(self, d: int, hidden_dim: int, mu:float, sigma: float, ts: torch.Tensor=None):

        super().__init__()
        self.d = d
        self.mu = mu
        self.sigma = sigma

        self.net_dgm = Net_DGM(d, hidden_dim, activation='Tanh')
        self.ts = ts

    def fit(self, max_updates: int, batch_size: int, option, device):

        optimizer = torch.optim.Adam(self.net_dgm.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)
        loss_fn = nn.MSELoss()
        
        pbar = tqdm.tqdm(total=max_updates)
        for it in range(max_updates):
            optimizer.zero_grad()

            input_domain = 0.5 + 2*torch.rand(batch_size, self.d, device=device, requires_grad=True)
            t0, T = self.ts[0], self.ts[-1]
            t = t0 + T*torch.rand(batch_size, 1, device=device, requires_grad=True)
            u_of_tx = self.net_dgm(t, input_domain)
            grad_u_x = get_gradient(u_of_tx,input_domain)
            grad_u_t = get_gradient(u_of_tx, t)
            laplacian = get_laplacian(grad_u_x, input_domain)
            target_functional = torch.zeros_like(u_of_tx)
            pde = grad_u_t + torch.sum(self.mu*input_domain.detach()*grad_u_x,1,keepdim=True) + 0.5*self.sigma**2 * laplacian - self.mu * u_of_tx
            MSE_functional = loss_fn(pde, target_functional)
            
            input_terminal = 0.5 + 2*torch.rand(batch_size, self.d, device=device, requires_grad=True)
            t = torch.ones(batch_size, 1, device=device) * T
            u_of_tx = self.net_dgm(t, input_terminal)
            target_terminal = option.payoff(input_terminal)
            MSE_terminal = loss_fn(u_of_tx, target_terminal)

            loss = MSE_functional + MSE_terminal
            loss.backward()
            optimizer.step()
            scheduler.step()
            if it%10 == 0:
                pbar.update(10)
                pbar.write("Iteration: {}/{}\t MSE functional: {:.4f}\t MSE terminal: {:.4f}\t Total Loss: {:.4f}".format(it, max_updates, MSE_functional.item(), MSE_terminal.item(), loss.item()))

    def sdeint(self, ts, x0, antithetic = False):
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
        if antithetic:
            x = torch.cat([x0.unsqueeze(1), x0.unsqueeze(1)], dim=0)
        else:
            x = x0.unsqueeze(1)
        batch_size = x.shape[0]
        device = x.device
        brownian_increments = torch.zeros(batch_size, len(ts), self.d, device=device)
        for idx, t in enumerate(ts[1:]):
            h = ts[idx+1]-ts[idx]
            if antithetic:
                brownian_increments[:batch_size//2,idx,:] = torch.randn(batch_size//2, self.d, device=device)*torch.sqrt(h)
                brownian_increments[-batch_size//2:,idx,:] = -brownian_increments[:batch_size//2, idx, :].clone()
            else:
                brownian_increments[:,idx,:] = torch.randn(batch_size, self.d, device=device)*torch.sqrt(h)
            x_new = x[:,-1,:] + self.mu*x[:,-1,:]*h + self.sigma*x[:,-1,:]*brownian_increments[:,idx,:]
            x = torch.cat([x, x_new.unsqueeze(1)],1)
        return x, brownian_increments

    
    def unbiased_price(self, ts: torch.Tensor, x0:torch.Tensor, option: BaseOption, MC_samples: int, ):
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
        with torch.no_grad():
            x, brownian_increments = self.sdeint(ts, x0)
        payoff = option.payoff(x[:,-1,:]) # (batch_size, 1)
        device = x.device
        batch_size = x.shape[0]
        t = ts.reshape(1,-1,1).repeat(batch_size,1,1)
        tx = torch.cat([t,x],2) # (batch_size, L, dim+1)
        Z = []
        for idt, t in enumerate(ts[:-1]):
            tt = torch.ones(batch_size, 1, device=device)*t
            tt.requires_grad_(True)
            xx = x[:,idt,:].requires_grad_(True)
            Y = self.net_dgm(tt, xx)
            Z.append(get_gradient(Y,xx))
        Z = torch.stack(Z, 1)

        stoch_int = 0
        for idx,t in enumerate(ts[:-1]):
            discount_factor = torch.exp(-self.mu*t)
            stoch_int += discount_factor * torch.sum(Z[:,idx,:]*brownian_increments[:,idx,:], 1, keepdim=True)
        mc = torch.exp(-self.mu*ts[-1])*payoff
        cv = stoch_int
        cv_mult = torch.mean((mc-mc.mean())*(cv-cv.mean())) / cv.var() # optimal multiplier. cf. Belomestny book
        return mc, mc - cv_mult*stoch_int # stoch_int has expected value 0, thus it doesn't add any bias to the MC estimator, and it is correlated with payoff


