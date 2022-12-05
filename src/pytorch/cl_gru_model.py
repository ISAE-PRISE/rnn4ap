# ---------------------------------------------------------------------
# RNN4AP project
# Copyright (C) 2021-2022 ISAE
# 
# Purpose:
# Evaluation of Recurrent Neural Networks for future Autopilot Systems
# 
# Contact:
# jean-baptiste.chaudron@isae-supaero.fr
# ---------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable
import math

# Pytorch implementation taken from excellent post
# https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

class cl_gru_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(cl_gru_model, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim

        # GRU layers (original from Pytorch)
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, dropout=0.2)

        # Fully connected layer (MLP) / No activation function is used
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().cuda()

        # Forward propagation by passing in the input and hidden state into the model
        out, _ = self.gru(x, h0.detach())

        # Reshaping the outputs in the shape of (batch_size, seq_length, hidden_size)
        # so that it can fit into the fully connected layer
        out = out[:, -1, :]

        # Convert the final state to our desired output shape (batch_size, output_dim)
        out = self.fc(out)

        return out
