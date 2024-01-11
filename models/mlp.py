import torch
import torch.nn as nn
import torch.nn.functional as F

# Hint
# output = F.relu(input)
# output = F.log_softmax(input)

__all__ = ['MLP']


# model structure: Input -> FC (ReLU) -> FC (Softmax) -> Output
class MLP_class(nn.Module):
    def __init__(self, input_size, hidden_dim, output_class):
        super().__init__()
        # TODO struct the model

    def forward(self, x):
        # Is this reshape function is necessary? Why?
        x = x.view(x.size(0), -1)
        # TODO struct the forward computation based on the its methods
        return x

def MLP(input_size, hidden_dim, output_class):
    return MLP_class(input_size, hidden_dim, output_class)
