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
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_class)

    def forward(self, x):
        # Is this reshape function is necessary? Why?
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x))
        return x

def MLP(input_size, hidden_dim, output_class):
    return MLP_class(input_size, hidden_dim, output_class)
