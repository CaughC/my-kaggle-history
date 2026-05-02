import torch
import torch.nn as nn

class TabularMLP(nn.Module):
    """A generic Multi-Layer Perceptron for tabular data."""
    def __init__(self, input_dim, layer_sizes, dropout=0.2, output_dim=1, activation="sigmoid"):
        super(TabularMLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for size in layer_sizes:
            layers.append(nn.Linear(prev_dim, size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = size
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if activation == "sigmoid":
            layers.append(nn.Sigmoid())
        elif activation == "softmax":
            layers.append(nn.Softmax(dim=1))
        # For regression (None), no final activation is added
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
