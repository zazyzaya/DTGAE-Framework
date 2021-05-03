import torch 
from torch import nn 

'''
GRU Class; very simple and lightweight
'''
class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)
        
        self.z_dim = z_dim 

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)
        
        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)
        
        if not include_h:
            return self.lin(xs)
        
        return self.lin(xs), h

'''
Slightly more complex RNN, but about equal at most tasks, though 
some papers show that LSTM is better in some instances than GRU

Best practice to use LSTM first, and if GRU performs as well to switch to that
'''
class LSTM(GRU):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(LSTM, self).__init__(x_dim, h_dim, z_dim, hidden_units=hidden_units)

        # Just swapping out one component with another
        self.rnn = nn.LSTM(
            x_dim, h_dim, num_layers=hidden_units
        )