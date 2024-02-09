import torch
import torch.nn as nn

class PositionalEncoder(nn.Module):
    """
    This torch Module define the positional encoding strategy as described in the NeRF official paper
    """
    def __init__(self,l ,log_sampling=True, include_input=True, embed=True):
        """

        :param l:
        :param log_sampling:
        :param embed:
        """
        super(PositionalEncoder, self).__init__()
        self.l = l

        if log_sampling:
            self.frequencies = 2.**torch.linspace(0., l - 1, l)
        else:
            self.frequencies = torch.linspace(2**0, 2**(l-1), l)

        self.encoding_functions = []
        if embed:
            self.functions = [lambda x: torch.cos(x), lambda x: torch.sin(x)]
            if include_input:
                self.functions.append(lambda x: x)
        else:
            self.functions = [lambda x: x]

        for freq in self.frequencies:
            for function in self.functions:
                self.encoding_functions.append(lambda x: function(x*freq*torch.pi))

    def forward(self,x):

        #Apply positional encoding to the input x -> [cos(2**0*pi*x),sin(2**0*pi*x),...,cos(2**(L-1)*pi*x),sin(2**(L-1)*pi*x)]
        return torch.cat([encoding_function(x) for encoding_function in self.encoding_functions],dim=-1)
