
import torch
from dataclasses import dataclass
from abc import abstractmethod
from typing import List

@dataclass
class BaseFinal:
    pass
    
    def __call__(self, x: torch.Tensor, **kwargs):
        raise NotImplementedError



class Bell(BaseFinal):
    
    def __init__(self):
        pass

    def __call__(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Tensor of shape (batch_size, d) where N is path length
        Returns
        -------
        output: torch.Tensor
            Tensor of shape (batch_size,1)
        """
        output = torch.exp(-torch.norm(x, p=2, dim=1, keepdim=True)**2)
        return output # (batch_size, 1)
