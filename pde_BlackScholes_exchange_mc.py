import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math
import pandas as pd

from lib.bsde_risk_neutral_measure import FBSDE_BlackScholes as FBSDE
from lib.options import Exchange
from lib.utils import set_seed


def sample_x0(batch_size, dim, device, lognormal: bool = True):
    if lognormal:
        sigma = 0.3
        mu = 0.08
        tau = 0.1
        z = torch.randn(batch_size, dim, device=device)
        x0 = torch.exp((mu-0.5*sigma**2)*tau + 0.3*math.sqrt(tau)*z) # lognormal
    else:
        x0 = torch.ones(batch_size, dim, device=device)
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
        device
        ):
    
    logfile = os.path.join(base_dir, "log.txt")
    ts = torch.linspace(0,T,n_steps+1, device=device)
    option = Exchange()
    fbsde = FBSDE(d=d, mu=mu, sigma=sigma, ffn_hidden=ffn_hidden, ts=ts, net_per_timestep=True)
    fbsde.to(device)
    
    x0 = sample_x0(1, d, device, lognormal=False)
    fbsde.eval()
    discounted_payoff = fbsde.unbiased_price_mc(ts=ts, x0=x0, option=option, MC_samples=10000)
    discounted_payoff_antithetic = fbsde.unbiased_price_mc(ts=ts, x0=x0, option=option, MC_samples=5000, antithetic=True)
    variance_red_factor = discounted_payoff.var() / discounted_payoff_antithetic.var()
    results = {'discounted_payoff':discounted_payoff.mean().item(), 
            'discounted_payoff_antithetic':discounted_payoff_antithetic.mean().item(),
            'variance_red_factor':variance_red_factor.item(),
            'var_discounted_payoff':discounted_payoff.var().item(),
            'var_discounted_payoff_antithetic':discounted_payoff_antithetic.var().item()}
    pd.DataFrame(results, index=[0]).to_csv(os.path.join(base_dir, 'results_antithetic.csv'))




        

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
    parser.add_argument('--T', default=0.5, type=float)
    parser.add_argument('--n_steps', default=50, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.3, type=float, help="risk free rate")
    
    parser.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    for i in range(args.n_seeds):
        seed = args.seed + i
        set_seed(seed)
        results_path = os.path.join(args.base_dir, "BS", "exchange_{}".format(args.d), "antithetic", "seed{}".format(seed))
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
            device=device
            )
    
