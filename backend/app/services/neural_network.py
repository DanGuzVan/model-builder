import torch
import torch.nn as nn
from typing import List, Dict, Any


class NeuralNetwork(nn.Module):
    """Configurable neural network for classification tasks."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        dropout: float = 0.0,
    ):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout_rate = dropout

        layers = []
        prev_size = input_size

        # Build hidden layers
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "NeuralNetwork":
        """Create a neural network from a configuration dictionary."""
        return cls(
            input_size=config["input_size"],
            hidden_layers=config["hidden_layers"],
            output_size=config["output_size"],
            dropout=config.get("dropout", 0.0),
        )

    def to_config(self) -> Dict[str, Any]:
        """Export network configuration to dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
        }

    def count_parameters(self) -> int:
        """Count the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_network_from_config(config: Dict[str, Any]) -> NeuralNetwork:
    """Factory function to create a neural network from config."""
    return NeuralNetwork.from_config(config)
