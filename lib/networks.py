import torch
import torch.nn as nn
from collections import namedtuple
from typing import Tuple

class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()

        layers = []
        for j in range(len(sizes)-1):
            layers.append(nn.Linear(sizes[j], sizes[j+1]))
            if j<(len(sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True

    def forward(self, x):
        return self.net(x)


class FFN_net_per_timestep(nn.Module):

    def __init__(self, sizes, ts: torch.Tensor, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.net = nn.ModuleList([FFN(sizes, activation, output_activation) for t in ts])

    def forward(self, x, idt = None):

        if x.dim() == 3: # x is of shape (batch_size, L, d)
            Y = []
            for i in range(x.shape[1]):
                Y.append(self.net[i](x[:,i,:]))
            Y = torch.stack(Y, 1)
        elif x.dim() == 2 and idt is not None:
            Y = self.net[idt](x)
        else:
            raise ValueError('idt needs to be passed')
        return Y


class RNN(nn.Module):

    def __init__(self, rnn_in, rnn_hidden, ffn_sizes, activation=nn.ReLU, output_activation=nn.Identity):
        super().__init__()
        self.rnn = nn.LSTM(input_size=rnn_in, hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True)
        layers = []
        for j in range(len(ffn_sizes)-1):
            layers.append(nn.Linear(ffn_sizes[j], ffn_sizes[j+1]))
            if j<(len(ffn_sizes)-2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.ffn = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad=False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad=True
            
    def forward(self, x):
        """Forward method 
        
        Parameters
        ----------
        x : torch.Tensor
            Sequential input. Tensor of size (N,L,d) where N is batch size, L is lenght of the sequence, and d is dimension of the path
        Returns
        -------
        output : torch.Tensor
            Sequential output. Tensor of size (N, L, d_out) containing the output from the last layer of the RNN for each timestep
        """
        output_RNN, _ = self.rnn(x)
        output = self.ffn(output_RNN)
        return output
