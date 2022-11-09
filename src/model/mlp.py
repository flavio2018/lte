"""MLP code from https://gist.github.com/KeAWang/10ac4a623f8e969795d79499f611c44e"""

from collections import OrderedDict

import torch
from torch import Tensor, Size
from torch.nn import Linear


class MLP(torch.nn.Sequential):
    """Multi-layered perception, i.e. fully-connected neural network
    Args:
        depth: number of hidden layers. 0 corresponds to a linear network
        input_width: dimensionality of inputs
        hidden_width: dimensionality of hidden layers
        output_width: dimensionality of final output
        activation: a torch.nn activation function
    """

    def __init__(
        self,
        depth: int,
        input_width: int,
        hidden_width: int,
        output_width: int,
        activation: str = "ReLU",
    ):
        self.depth = depth
        self.input_width = input_width
        self.hidden_width = hidden_width
        self.output_width = output_width
        self.activation = activation

        modules = []
        if depth == 0:
            modules.append(("linear1", Linear(input_width, output_width)))
        else:
            modules.append(("linear1", Linear(input_width, hidden_width)))
            for i in range(1, depth + 1):
                modules.append((f"{activation}{i}", getattr(torch.nn, activation)()))
                modules.append(
                    (
                        f"linear{i + 1}",
                        Linear(
                            hidden_width, hidden_width if i != depth else output_width
                        ),
                    )
                )
        modules = OrderedDict(modules)
        super().__init__(modules)
