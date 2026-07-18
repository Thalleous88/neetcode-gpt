import torch
import torch.nn as nn
import math
from typing import List


class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        xavier = math.sqrt(2/(fan_in+fan_out))

        res = torch.randn(fan_out, fan_in) * xavier

        res = torch.round(res, decimals=4)

        return res.tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        torch.manual_seed(0)
        kaiming = math.sqrt(2/fan_in)

        res = torch.randn(fan_out, fan_in) * kaiming

        res = torch.round(res, decimals=4)

        return res.tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        torch.manual_seed(0)
        weights = []
        stds = []

        for layer in range(num_layers):
            fan_in = input_dim if layer == 0 else hidden_dim

            if init_type == "xavier":
                std = math.sqrt(2 / (fan_in + hidden_dim))
                weights.append(torch.randn(hidden_dim, fan_in) * std)
            elif init_type == "kaiming":
                std = math.sqrt(2 / fan_in)
                weights.append(torch.randn(hidden_dim, fan_in) * std)
            else:
                weights.append(torch.randn(hidden_dim, fan_in))

        inp = torch.randn(input_dim)

        for w in weights:
            inp = torch.relu(torch.nn.functional.linear(inp, w))
            stds.append(round(inp.std().item(), 2))

        return stds

        

