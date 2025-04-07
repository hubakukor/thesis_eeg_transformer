import torch
import torch.nn as nn
import numpy as np
import math


class SinusoidalPositionalEmbedding(nn.Module):
    '''
    Sinusoidal Positional Embedding
    The type of embedding used in the original transformer model (Attention is all you need)
    '''
    def __init__(self, seq_len, d_model):
        super().__init__()

        # Create a matrix of shape [max_len, d_model]
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term) # even dimensions
        if d_model % 2 == 0:
            pe[:, 1::2] = torch.cos(position * div_term) # odd dimensions
        else:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :].unsqueeze(1)