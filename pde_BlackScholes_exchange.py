import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
import math

from lib.bsde_risk_neutral_measure import FBSDE_BlackScholes as FBSDE
from lib.options import Exchange


def sample_x0(batch_size, dim, device, lognormal: Bool = True):
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

def 

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
    option = Exchange()
    fbsde = FBSDE(d, mu, sigma, ffn_hidden)
    fbsde.to(device)
    optimizer = torch.optim.RMSprop(fbsde.parameters(), lr=0.0005)
    
    pbar = tqdm.tqdm(total=max_updates)
    losses = []
    for idx in range(max_updates):
        optimizer.zero_grad()
        x0 = sample_x0(batch_size, d, device, lognormal=True)
        if method=="bsde":
            loss, _, _ = fbsde.bsdeint(ts=ts, x0=x0, option=option)
        else:
            loss, _, _ = fbsde.conditional_expectation(ts=ts, x0=x0, option=option)
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().item())
        # testing
        if idx%10 == 0:
            with torch.no_grad():
                x0 = torch.ones(5000,d,device=device) # we do monte carlo
                loss, Y, payoff = fbsde.bsdeint(ts=ts,x0=x0,option=option)
                payoff = torch.exp(-mu * ts[-1]) * payoff.mean()
            
            pbar.update(10)
            write("loss={:.4f}, Monte Carlo price={:.4f}, predicted={:.4f}".format(loss.item(),payoff.item(), Y[0,0,0].item()),logfile,pbar)
    
    result = {"state":fbsde.state_dict(),
            "loss":losses}

    torch.save(result, os.path.join(base_dir, "result.pth.tar"))


def visualize(T,
        n_steps,
        d,
        mu,
        sigma,
        ffn_hidden,
        base_dir,
        ):
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import types
    assert d==2, "visualization is only implemented for 2-dimensional PDE"
    ts = torch.linspace(0,T,n_steps+1, device=device)
    option = Exchange()
    fbsde = FBSDE(d, mu, sigma, ffn_hidden)
    checkpoint = torch.load(os.path.join(base_dir, "result.pth.tar"), map_location="cpu")
    fbsde.load_state_dict(checkpoint["state"])

    with torch.no_grad():
        x0 = torch.linspace(0.6,1.4,500)
        x1 = torch.linspace(0.6,1.4,500)
        X0,X1 = torch.meshgrid([x0,x1])
        X = torch.cat([X0.reshape(-1,1), X1.reshape(-1,1)],1)
        t_coarse = ts[::n_steps//10]
        X = X.unsqueeze(1).repeat(1,len(t_coarse),1)
        t = t_coarse.reshape(1,-1,1).repeat(X.shape[0],1,1)
        tx = torch.cat([t,X],2)
        Y = fbsde.f(tx)
    ims = []
    fig = plt.figure()
    X0 = X0.numpy()
    X1 = X1.numpy()
    for idx, t in enumerate(t_coarse):
        Z = Y[:,idx,:].numpy().reshape(500,500) 
        im = plt.contourf(X0,X1,Z,levels=80)
        ims.append(im.collections)
        #plt.savefig(os.path.join(base_dir, "contourf{}.png".format(idx)))
    anim = animation.ArtistAnimation(fig, ims, interval=400, repeat_delay=3000)
    anim.save(os.path.join(base_dir, "contourf.mp4")) 
    anim.save(os.path.join(base_dir, "contourf.gif"), dpi=80, writer='imagemagick') 
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--base_dir', default='./numerical_results/', type=str)
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--d', default=2, type=int)
    parser.add_argument('--max_updates', default=5000, type=int)
    parser.add_argument('--ffn_hidden', default=[20,20,20], nargs="+", type=int, help="hidden sizes of ffn networks approximations")
    parser.add_argument('--T', default=1, type=float)
    parser.add_argument('--n_steps', default=100, type=int, help="number of steps in time discrretisation")
    parser.add_argument('--mu', default=0.05, type=float, help="risk free rate")
    parser.add_argument('--sigma', default=0.3, type=float, help="risk free rate")
    parser.add_argument('--method', default="bsde", type=str, help="learning method", choices=["bsde","l2-proj"])
    
    parser.add_argument('--visualize', action='store_true', default=False)

    args = parser.parse_args()
    
    if torch.cuda.is_available() and args.use_cuda:
        device = "cuda:{}".format(args.device)
    else:
        device="cpu"
    
    results_path = os.path.join(args.base_dir, "BS", "exchange_{}".format(args.d), args.method)
    if not os.path.exists(results_path):
        os.makedirs(results_path)

    if args.visualize:
        visualize(T=args.T,
           n_steps=args.n_steps,
           d=args.d,
           mu=args.mu,
           sigma=args.sigma,
           ffn_hidden=args.ffn_hidden,
           base_dir=results_path)
    else:
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
