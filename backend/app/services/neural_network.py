"""
Neural Network Module for Configurable Classification.

This module provides a flexible MLP architecture for tabular classification tasks,
following best practices from neural network research:
- ReLU activation for hidden layers (addressing vanishing gradient problem)
- Dropout for regularization (Srivastava et al., 2014)
- He initialization for weights (He et al., 2015)
- CrossEntropyLoss expects raw logits (numerically stable)

Author: NN Optimizer Project
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple


class ConfigurableClassifier(nn.Module):
    """
    Flexible MLP for tabular classification.
    
    Architecture: Input -> [Linear + ReLU + Dropout]* -> Linear -> Output (logits)
    
    Scientific basis:
    - ReLU activation prevents vanishing gradients in deep networks
    - Dropout (Srivastava et al., 2014) provides regularization
    - He initialization (He et al., 2015) optimal for ReLU networks
    
    Args:
        input_size: Number of input features
        hidden_layers: List of hidden layer sizes, e.g., [64, 32]
        output_size: Number of output classes
        dropout: Dropout rate between layers (default 0.2, research suggests 0.2-0.5)
        activation: Activation function to use (default: 'relu')
    
    Example:
        >>> model = ConfigurableClassifier(10, [64, 32], 3, dropout=0.3)
        >>> x = torch.randn(32, 10)  # batch of 32 samples
        >>> logits = model(x)  # shape: (32, 3)
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super(ConfigurableClassifier, self).__init__()
        
        # Store configuration for serialization
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.dropout_rate = dropout
        self.activation_name = activation
        
        # Validate inputs
        if input_size <= 0:
            raise ValueError(f"input_size must be positive, got {input_size}")
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        if not hidden_layers:
            raise ValueError("hidden_layers cannot be empty")
        if not 0 <= dropout < 1:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        
        # Build network layers dynamically
        layers = []
        prev_size = input_size
        
        # Get activation function
        activation_fn = self._get_activation(activation)
        
        # Build hidden layers: Linear -> Activation -> Dropout
        for i, hidden_size in enumerate(hidden_layers):
            if hidden_size <= 0:
                raise ValueError(f"Hidden layer {i} size must be positive, got {hidden_size}")
            
            # Linear layer
            linear = nn.Linear(prev_size, hidden_size)
            # He initialization for ReLU
            nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
            nn.init.zeros_(linear.bias)
            layers.append(linear)
            
            # Activation
            layers.append(activation_fn())
            
            # Dropout (skip if rate is 0)
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            
            prev_size = hidden_size
        
        # Output layer (no activation - CrossEntropyLoss handles softmax)
        output_layer = nn.Linear(prev_size, output_size)
        nn.init.xavier_uniform_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        # Combine into sequential
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str):
        """Get activation function class by name."""
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'elu': nn.ELU,
            'selu': nn.SELU,
            'gelu': nn.GELU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from: {list(activations.keys())}")
        return activations[name.lower()]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Logits tensor of shape (batch_size, output_size)
            Note: Returns raw logits, use CrossEntropyLoss for training
        """
        return self.network(x)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions (argmax of logits).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.argmax(logits, dim=1)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class probabilities (softmax of logits).
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
        
        Returns:
            Probability tensor of shape (batch_size, output_size)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ConfigurableClassifier":
        """
        Create a network from a configuration dictionary.
        
        Args:
            config: Dict with keys 'input_size', 'hidden_layers', 'output_size', 
                   and optionally 'dropout', 'activation'
        
        Returns:
            Configured network instance
        """
        return cls(
            input_size=config["input_size"],
            hidden_layers=config["hidden_layers"],
            output_size=config["output_size"],
            dropout=config.get("dropout", 0.2),
            activation=config.get("activation", "relu"),
        )
    
    def to_config(self) -> Dict[str, Any]:
        """
        Export network configuration to dictionary.
        
        Returns:
            Configuration dict that can recreate this network
        """
        return {
            "input_size": self.input_size,
            "hidden_layers": self.hidden_layers,
            "output_size": self.output_size,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
        }
    
    def count_parameters(self, trainable_only: bool = True) -> int:
        """
        Count the total number of parameters.
        
        Args:
            trainable_only: If True, count only trainable parameters
        
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


# Alias for backward compatibility with existing code
NeuralNetwork = ConfigurableClassifier


def create_network(config: Dict[str, Any]) -> ConfigurableClassifier:
    """
    Factory function to create network from config dict.
    
    Args:
        config: Configuration dictionary with network parameters
    
    Returns:
        Configured ConfigurableClassifier instance
    
    Example:
        >>> config = {'input_size': 10, 'hidden_layers': [64, 32], 
        ...           'output_size': 3, 'dropout': 0.2}
        >>> model = create_network(config)
    """
    return ConfigurableClassifier.from_config(config)


def create_network_from_config(config: Dict[str, Any]) -> ConfigurableClassifier:
    """Alias for create_network (backward compatibility)."""
    return create_network(config)


def get_network_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Return detailed summary of network architecture.
    
    Args:
        model: PyTorch model to summarize
    
    Returns:
        Dict containing:
        - total_params: Total parameter count
        - trainable_params: Trainable parameter count
        - layers: List of layer info dicts
        - input_size: Input size (if ConfigurableClassifier)
        - output_size: Output size (if ConfigurableClassifier)
        - architecture: Human-readable architecture string
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    layers_info = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.Dropout, nn.BatchNorm1d)):
            layer_info = {
                "name": name,
                "type": module.__class__.__name__,
                "params": sum(p.numel() for p in module.parameters()),
            }
            
            if isinstance(module, nn.Linear):
                layer_info["in_features"] = module.in_features
                layer_info["out_features"] = module.out_features
            elif isinstance(module, nn.Dropout):
                layer_info["p"] = module.p
            
            layers_info.append(layer_info)
    
    summary = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": total_params - trainable_params,
        "layers": layers_info,
        "num_layers": len([l for l in layers_info if l["type"] == "Linear"]),
    }
    
    # Add specific info for ConfigurableClassifier
    if isinstance(model, ConfigurableClassifier):
        summary["input_size"] = model.input_size
        summary["output_size"] = model.output_size
        summary["hidden_layers"] = model.hidden_layers
        summary["dropout_rate"] = model.dropout_rate
        
        # Build architecture string
        arch_parts = [str(model.input_size)]
        arch_parts.extend(str(h) for h in model.hidden_layers)
        arch_parts.append(str(model.output_size))
        summary["architecture"] = " -> ".join(arch_parts)
    
    return summary


def calculate_model_memory(model: nn.Module, input_shape: Tuple[int, ...], 
                           dtype: torch.dtype = torch.float32) -> Dict[str, float]:
    """
    Estimate memory usage of model.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (without batch dimension)
        dtype: Data type for calculations
    
    Returns:
        Dict with memory estimates in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    # Estimate activation memory (rough)
    # This is a simplified estimate
    input_size = 1
    for dim in input_shape:
        input_size *= dim
    
    bytes_per_element = torch.tensor([], dtype=dtype).element_size()
    activation_estimate = input_size * bytes_per_element * 4  # rough multiplier
    
    return {
        "parameters_mb": param_size / (1024 * 1024),
        "buffers_mb": buffer_size / (1024 * 1024),
        "activation_estimate_mb": activation_estimate / (1024 * 1024),
        "total_estimate_mb": (param_size + buffer_size + activation_estimate) / (1024 * 1024),
    }