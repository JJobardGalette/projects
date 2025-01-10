import torch
import torch.nn as nn

class QNeural(nn.Module):
    def __init__(self, input_size, output_size):
        self.input_layer = nn.Linear(input_size, 64)
        self.hidden_layer = nn.Linear(64, 64)
        self.output_layer = nn.Linear(64, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        x = self.activation(self.hidden_layer(x))
        return self.output_layer(x)
        