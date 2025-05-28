import torch
import torch.nn as nn
import numpy as np
import math


class SinusoidalPositionalEmbedding(nn.Module):
    '''
    Sinusoidal Positional Embedding
    The type of embedding used in the original transformer model (Attention is all you need)

    Args:
        seq_len: Length of the sequence.
        d_model: Dimensionality of the model.

    Returns:
        pe: Positional embedding matrix.
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



class LearnablePositionalEmbedding(nn.Module):
    '''
    Learnable Positional Embedding module.
    source: https://medium.com/biased-algorithms/how-to-modify-positional-encoding-in-torch-nn-transformer-bf7f5c5ba9c3

    Args:
        seq_len: Length of the sequence.
        d_model: Dimensionality of the model.

    Returns:
        pos_embedding: Learnable positional embedding matrix.
    '''

    def __init__(self, seq_len, d_model):
        super(LearnablePositionalEmbedding, self).__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(seq_len, d_model))
        nn.init.xavier_uniform_(self.pos_embedding)  # Initialize the learnable parameters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pos_embedding[:x.size(0), :].unsqueeze(1)  # [seq_len, 1, d_model]