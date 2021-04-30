import torch
from torch import nn 

'''
Very generic module to hold two 
'''
class DTGAE(nn.Module):
    def __init__(self, encoder, rnn):
        self.encoder = encoder 
        self.rnn = rnn