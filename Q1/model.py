#!/usr/bin/env python
# coding: utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining FNNModel
class FNNModel(nn.Module):
    """Container module with an encoder, a feedforward module, and a decoder."""

    def __init__(self, vocab_size, input_dim, hidden_dim, context_size, dropout, tie_weights=False):
        super(FNNModel, self).__init__() # Inherited from the parent class nn.Module
        self.vocab_size = vocab_size # number of tokens in the corpus dictionary
        self.context_size = context_size
        self.input_dim = input_dim
        self.drop = dropout
        
        # vocab_size - vocab, input_dim - dimensionality of the embeddings
        self.encoder = nn.Embedding(vocab_size, input_dim) # used to store word embeddings and retrieve them using indices
        
        # Declaring the layers
        self.input = nn.Linear(context_size * input_dim, hidden_dim) # linear layer (input)
        self.hidden = nn.Tanh() # Second layer - tahn activation layer (non-linearity layer)
        self.decoder = nn.Linear(hidden_dim, vocab_size, bias = False ) # decoder - linearity layer\
        
        if tie_weights:
            if hidden_dim != input_size:
                raise ValueError(
                    'When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.nhid = hidden_dim
       

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.weight)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, input): # Forward pass: stacking each layer together 
        emb = self.encoder(input).view((-1,self.context_size*self.input_dim))
        x = self.input(emb) 
        output = self.hidden(x) # applying tanh
        decoded = self.decoder(output)
        decoded = decoded.view(-1, self.vocab_size)
        log_probs = F.log_softmax(decoded, dim=1) # applies log after softmax - output
        return log_probs 

    
#     def init_hidden(self, bsz):
#         weight = next(self.parameters())
    
# Positional Encoding
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
