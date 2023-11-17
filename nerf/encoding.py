import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    This torch Module define the positional encoding strategy as described in the NeRF official paper
    """
    def __init__(self,L):
        super(PositionalEncoder, self).__init__()
        self.L = L
        self.freqs = torch.linspace(2**0,2**(L-1),L)
        self.encoding_functions = []

        for freq in self.freqs:
            self.encoding_functions.append(lambda x: torch.cos(freq*torch.pi*x))
            self.encoding_functions.append(lambda x: torch.sin(freq*torch.pi*x))

    def forward(self,x):
        #Apply positional encoding to the input x -> [cos(2**0*pi*x),sin(2**0*pi*x),...,cos(2**(L-1)*pi*x),sin(2**(L-1)*pi*x)]
        return torch.cat([encoding_function(x) for encoding_function in self.encoding_functions],dim=-1)
