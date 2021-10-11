import torch
import numpy as np


def write(msg, logfile, pbar):
    pbar.write(msg)
    with open(logfile, "a") as f:
        f.write(msg)
        f.write("\n")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)



