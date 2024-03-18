import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, n_attributes: int, n_actions: int, inner_layers_neuron_numbers: list[int]):
        super().__init__()

        inner_layers_neuron_numbers.append(n_actions)
        inner_layers_neuron_numbers.insert(0, n_attributes)

        ilnn = inner_layers_neuron_numbers.copy()
        self.linear_layers = [nn.Linear(in_features=ilnn[i], out_features=ilnn[i+1]) for i in range(len(ilnn)-1)]

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        t = state_tensor
        for linear_layer in self.linear_layers:
            t = F.relu(linear_layer(t))
        return t