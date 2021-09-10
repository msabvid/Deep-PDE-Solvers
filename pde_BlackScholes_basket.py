import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import pandas as pd

from lib.bsde_risk_neutral_measure import FBSDE_BlackScholes as FBSDE
from lib.options import Basket
from lib.utils import set_seed


def sample_x0(batch_size, dim, device, lognormal: bool = True):
    if lognormal:
        sigma = 0.3
        mu = 0.08
        tau = 0.1
        z = torch.randn(batch_size, dim, device=device)
        x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    else:
        x0 = 0.7 * torch.ones(batch_size, dim, device=device)
    return x0
    

def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")


def train(T,
        n_steps,
        d,
        mu,
        sigma,
        ffn_hidden,
        max_updates,
        batch_size, 
        base_dir,
        device,
        method
        ):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    K = 0.7 * d
    option = Basket(K=K)

    fbsde = FBSDE(d, mu, sigma, ffn_hidden)
    fbsde.to(device)
    optimizer = torch.optim.RMSprop(fbsde.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = (10000,),gamma=0.1)

    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device, lognormal=False)
        if method=="bsde":
            loss, _, _ = fbsde.bsdeint(ts=ts, x0=x0, option=option)
        else:
            loss, _, _ = fbsde.l2_proj(ts=ts, x0=x0, option=option)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        # testing
        if idx%10 == 0:
            with torch.no_grad():
                x0 = sample_x0(5000, d, device, lognormal=False)
                if method == 'bsde':
                    loss, Y, payoff = fbsde.bsdeint(ts=ts,x0=x0,option=option)
                elif method == 'l2_proj':
                    loss, Y, payoff = fbsde.l2_proj(ts=ts,x0=x0,option=option)
                payoff = torch.exp(-mu * ts[-1]) * payoff.mean()
            
            pbar.update(10)
            write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(),payoff.item(), Y[0,0,0].item()),logfile,pbar)
    
    
    
    x0 = sample_x0(1, d, device, lognormal=False)
    discounted_payoff, discounted_payoff_cv = fbsde.unbiased_price(ts=ts, x0=x0, option=option, MC_samples=10**6, method=method)
    variance_red_factor = discounted_payoff.var() / discounted_payoff_cv.var()
    results = {'discounted_payoff':discounted_payoff.mean().item(), 
            'discounted_payoff_cv':discounted_payoff_cv.mean().item(),
            'variance_red_factor':variance_red_factor.item(),
            'var_discounted_payoff':discounted_payoff.var().item(),
            'var_discounted_payoff_cv':discounted_payoff_cv.var().item()}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(base_dir, 'results.csv'))


    result = {"state":fbsde.state_dict(),
            "loss":losses}

    torch.save(result, os.path.join(base_dir, "result.pth.tar"))






if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--n_seeds', default=10, type=int)


    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20], nargs="+", type=int, help="hidden sizes of ffn networks approximations")
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--mu', default=0.5, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=1., type=float, help="risk free rate")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","l2_proj"])
    

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    for i in range(args.n_seeds):
        seed = args.seed + i
        set_seed(seed)
        results_path = os.path.join(args.base_dir, "BS", "exchange_{}".format(args.d), args.method, "seed{}".format(seed))
        if not os.path.exists(results_path):
            os.makedirs(results_path)

        train(T=args.T,
            n_steps=args.n_steps,
            d=args.d,
            mu=args.mu,
            sigma=args.sigma,
            ffn_hidden=args.ffn_hidden,
            max_updates=args.max_updates,
            batch_size=args.batch_size,
            base_dir=results_path,
            device=device,
            method=args.method
            )
