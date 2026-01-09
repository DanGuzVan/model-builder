"""
Particle Swarm Optimization for Neural Network Hyperparameter Tuning.

This module implements PSO (Kennedy & Eberhart, 1995) for optimizing
continuous hyperparameters like learning rate and dropout rate.

Scientific basis:
- PSO excels at continuous optimization problems
- Fast convergence compared to grid/random search
- Key parameters: inertia (w), cognitive (c1), social (c2)
- Log-scale search for learning rate (Bengio, 2012)

Author: NN Optimizer Project
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging

from app.config import settings
from app.database import SessionLocal
from app.models import OptimizationRun, Dataset, TrainingStatus
from app.services.neural_network import NeuralNetwork, ConfigurableClassifier
from app.services.trainer import TrainerService

logger = logging.getLogger(__name__)


@dataclass
class Particle:
    """
    Represents a particle in the PSO swarm.
    
    Each particle has:
    - position: Current hyperparameters in search space
    - velocity: Current movement direction/speed
    - best_position: Personal best found so far
    - best_fitness: Fitness at personal best
    - fitness: Current fitness value
    """
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray = field(default_factory=lambda: np.array([]))
    best_fitness: float = float("-inf")
    fitness: float = 0.0
    
    def __post_init__(self):
        """Initialize personal best to starting position."""
        if self.best_position.size == 0:
            self.best_position = self.position.copy()


@dataclass
class SearchSpace:
    """
    Defines the hyperparameter search space for PSO.
    
    Each dimension has bounds and optionally uses log scale.
    """
    names: List[str]
    bounds: np.ndarray  # Shape: (n_dims, 2) - [low, high] for each dimension
    log_scale: List[bool]  # Which dimensions use log scale
    
    @property
    def n_dims(self) -> int:
        return len(self.names)
    
    def clip(self, position: np.ndarray) -> np.ndarray:
        """Clip position to valid bounds."""
        return np.clip(position, self.bounds[:, 0], self.bounds[:, 1])
    
    def decode(self, position: np.ndarray) -> Dict[str, float]:
        """Convert position array to hyperparameter dict."""
        params = {}
        for i, name in enumerate(self.names):
            value = position[i]
            if self.log_scale[i]:
                # Convert from log scale
                value = 10 ** value
            params[name] = value
        return params
    
    def random_position(self) -> np.ndarray:
        """Generate random position within bounds."""
        return np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
    
    def random_velocity(self, scale: float = 0.1) -> np.ndarray:
        """Generate random initial velocity."""
        ranges = self.bounds[:, 1] - self.bounds[:, 0]
        return np.random.uniform(-ranges * scale, ranges * scale)


class PSOOptimizer:
    """
    Particle Swarm Optimization for neural network hyperparameters.
    
    Optimizes continuous hyperparameters using swarm intelligence.
    Each particle represents a candidate solution that moves through
    the search space influenced by its personal best and the global best.
    
    Search space:
    - learning_rate: [0.0001, 0.1] (log scale, stored as [-4, -1])
    - dropout: [0.0, 0.5] (linear scale)
    - hidden_size: [16, 256] (linear scale, rounded to int)
    
    PSO Update Equations:
    v_i(t+1) = w*v_i(t) + c1*r1*(pbest_i - x_i) + c2*r2*(gbest - x_i)
    x_i(t+1) = x_i(t) + v_i(t+1)
    
    Args:
        n_particles: Number of particles in swarm (default 15)
        n_iterations: Maximum iterations (default 10)
        w: Inertia weight - controls momentum (default 0.7)
        c1: Cognitive coefficient - attraction to personal best (default 1.5)
        c2: Social coefficient - attraction to global best (default 1.5)
        eval_epochs: Epochs for fitness evaluation (default 10)
    
    References:
        Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    """
    
    # Default search space configuration
    DEFAULT_SEARCH_SPACE = SearchSpace(
        names=["learning_rate", "dropout", "hidden_size"],
        bounds=np.array([
            [-4.0, -1.0],    # learning_rate in log10 scale: 10^-4 to 10^-1
            [0.0, 0.5],      # dropout rate
            [16.0, 256.0],   # hidden layer size
        ]),
        log_scale=[True, False, False]  # Only learning_rate uses log scale
    )
    
    def __init__(
        self,
        n_particles: int = 15,
        n_iterations: int = 10,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
        eval_epochs: int = 10,
        search_space: Optional[SearchSpace] = None
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.eval_epochs = eval_epochs
        self.search_space = search_space or self.DEFAULT_SEARCH_SPACE
        
        # State
        self.swarm: List[Particle] = []
        self.global_best_position: Optional[np.ndarray] = None
        self.global_best_fitness: float = float("-inf")
        self.history: List[Dict[str, Any]] = []
        
        # Device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _initialize_swarm(self) -> List[Particle]:
        """
        Create initial random particles within bounds.
        
        Returns:
            List of initialized Particle objects
        """
        particles = []
        for _ in range(self.n_particles):
            position = self.search_space.random_position()
            velocity = self.search_space.random_velocity(scale=0.1)
            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
                best_fitness=float("-inf"),
                fitness=0.0
            )
            particles.append(particle)
        return particles
    
    def _decode_position(self, position: np.ndarray) -> Dict[str, Any]:
        """
        Convert position array to hyperparameter dict.
        
        Args:
            position: Raw position in search space
            
        Returns:
            Dict with decoded hyperparameters
        """
        params = self.search_space.decode(position)
        
        # Round hidden_size to integer
        if "hidden_size" in params:
            params["hidden_size"] = int(round(params["hidden_size"]))
        
        return params
    
    def _evaluate(
        self,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_size: int,
        output_size: int
    ) -> float:
        """
        Train network with params and return validation accuracy.
        
        This is the fitness function for PSO. It creates a network
        with the given hyperparameters, trains it briefly, and
        returns the validation accuracy as fitness.
        
        Args:
            params: Decoded hyperparameters
            X_train, y_train: Training data
            X_val, y_val: Validation data
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            Validation accuracy (0.0 to 1.0), or 0.0 on error
        """
        try:
            # Build network config
            hidden_size = params.get("hidden_size", 64)
            dropout = params.get("dropout", 0.2)
            learning_rate = params.get("learning_rate", 0.001)
            
            # Create simple 2-layer network with optimized hidden size
            config = {
                "input_size": input_size,
                "hidden_layers": [hidden_size, hidden_size // 2],
                "output_size": output_size,
                "dropout": dropout,
            }
            
            model = ConfigurableClassifier.from_config(config).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            
            # Create data loader
            train_dataset = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            
            # Quick training
            model.train()
            for _ in range(self.eval_epochs):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            # Evaluate on validation set
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.from_numpy(X_val).float().to(self.device)
                y_val_tensor = torch.from_numpy(y_val).long().to(self.device)
                outputs = model(X_val_tensor)
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(y_val_tensor).sum().item() / len(y_val)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"PSO evaluation failed: {e}")
            return 0.0
    
    def _update_velocity(self, particle: Particle):
        """
        Update particle velocity using PSO equations.
        
        v(t+1) = w*v(t) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        
        Args:
            particle: Particle to update
        """
        r1 = np.random.random(self.search_space.n_dims)
        r2 = np.random.random(self.search_space.n_dims)
        
        # Cognitive component (attraction to personal best)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        
        # Social component (attraction to global best)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        
        # Update velocity
        particle.velocity = self.w * particle.velocity + cognitive + social
        
        # Velocity clamping to prevent explosion
        max_velocity = (self.search_space.bounds[:, 1] - self.search_space.bounds[:, 0]) * 0.2
        particle.velocity = np.clip(particle.velocity, -max_velocity, max_velocity)
    
    def _update_position(self, particle: Particle):
        """
        Update particle position and clip to bounds.
        
        x(t+1) = x(t) + v(t+1)
        
        Args:
            particle: Particle to update
        """
        particle.position = particle.position + particle.velocity
        particle.position = self.search_space.clip(particle.position)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_size: int,
        output_size: int,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Run PSO optimization.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            input_size: Number of input features
            output_size: Number of output classes
            callback: Optional callback called each iteration
            
        Returns:
            Dict containing:
            - best_params: Best hyperparameters found
            - best_fitness: Best validation accuracy
            - history: List of dicts with iteration stats
        """
        # Split data for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize swarm
        self.swarm = self._initialize_swarm()
        self.global_best_position = None
        self.global_best_fitness = float("-inf")
        self.history = []
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            fitness_values = []
            
            # Evaluate each particle
            for particle in self.swarm:
                params = self._decode_position(particle.position)
                particle.fitness = self._evaluate(
                    params, X_train, y_train, X_val, y_val,
                    input_size, output_size
                )
                fitness_values.append(particle.fitness)
                
                # Update personal best
                if particle.fitness > particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Update global best
                if particle.fitness > self.global_best_fitness:
                    self.global_best_fitness = particle.fitness
                    self.global_best_position = particle.position.copy()
            
            # Update velocities and positions
            for particle in self.swarm:
                self._update_velocity(particle)
                self._update_position(particle)
            
            # Record history
            iteration_stats = {
                "iteration": iteration + 1,
                "best_fitness": self.global_best_fitness,
                "avg_fitness": np.mean(fitness_values),
                "std_fitness": np.std(fitness_values),
                "progress": (iteration + 1) / self.n_iterations
            }
            self.history.append(iteration_stats)
            
            # Call callback if provided
            if callback:
                callback(iteration_stats)
            
            logger.info(
                f"PSO Iteration {iteration + 1}/{self.n_iterations}: "
                f"Best={self.global_best_fitness:.4f}, "
                f"Avg={np.mean(fitness_values):.4f}"
            )
        
        # Return results
        best_params = self._decode_position(self.global_best_position)
        return {
            "best_params": best_params,
            "best_fitness": self.global_best_fitness,
            "history": self.history,
            "network_config": {
                "input_size": input_size,
                "hidden_layers": [
                    best_params["hidden_size"],
                    best_params["hidden_size"] // 2
                ],
                "output_size": output_size,
                "dropout": best_params["dropout"],
            }
        }


class PSOService:
    """
    Service class for PSO optimization (integrates with FastAPI).
    
    This maintains backward compatibility with the existing API.
    """

    # Search space bounds (for backward compatibility)
    BOUNDS = {
        "num_layers": (1, 5),
        "layer_size": (16, 256),
        "dropout": (0.0, 0.5),
        "learning_rate": (-4, -1),  # log10 scale
    }

    @staticmethod
    def decode_position(position: Dict[str, float], input_size: int, output_size: int) -> Dict[str, Any]:
        """Decode particle position to network configuration."""
        num_layers = max(1, int(round(position["num_layers"])))
        layer_size = max(16, int(round(position["layer_size"])))
        dropout = max(0.0, min(0.5, position["dropout"]))
        learning_rate = 10 ** position["learning_rate"]

        hidden_layers = [layer_size] * num_layers

        return {
            "network_config": {
                "input_size": input_size,
                "hidden_layers": hidden_layers,
                "output_size": output_size,
                "dropout": dropout,
            },
            "learning_rate": learning_rate,
        }

    @staticmethod
    def evaluate_fitness(
        config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 20,
    ) -> float:
        """Evaluate fitness of a network configuration."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            model = NeuralNetwork.from_config(config["network_config"]).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

            train_dataset = TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            )
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

            # Quick training
            model.train()
            for _ in range(epochs):
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Evaluate
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.from_numpy(X_val).float().to(device)
                y_val_tensor = torch.from_numpy(y_val).long().to(device)
                outputs = model(X_val_tensor)
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(y_val_tensor).sum().item() / len(y_val)

            return accuracy

        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0

    @staticmethod
    def run_optimization(run_id: int):
        """Run PSO optimization (background task)."""
        db = SessionLocal()
        try:
            # Get optimization run
            opt_run = db.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()
            if not opt_run:
                return

            # Update status
            opt_run.status = TrainingStatus.RUNNING.value
            db.commit()

            # Get dataset
            dataset = db.query(Dataset).filter(Dataset.id == opt_run.dataset_id).first()
            if not dataset:
                opt_run.status = TrainingStatus.FAILED.value
                db.commit()
                return

            # Load data
            X, y = TrainerService.load_dataset(dataset.filename)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # PSO parameters
            config = opt_run.config
            num_particles = config.get("num_particles", 30)
            max_iterations = config.get("max_iterations", 100)
            w = config.get("w", 0.7)  # Inertia
            c1 = config.get("c1", 1.5)  # Cognitive
            c2 = config.get("c2", 1.5)  # Social

            # Initialize particles using dataclass
            particles = []
            for _ in range(num_particles):
                position = {}
                velocity = {}
                for key, (low, high) in PSOService.BOUNDS.items():
                    position[key] = np.random.uniform(low, high)
                    velocity[key] = np.random.uniform(-(high - low), high - low) * 0.1
                particles.append({
                    "position": position,
                    "velocity": velocity,
                    "best_position": position.copy(),
                    "best_fitness": float("-inf"),
                })
            
            global_best_position = None
            global_best_fitness = float("-inf")

            history = []

            # PSO loop
            for iteration in range(max_iterations):
                # Check if cancelled
                db.refresh(opt_run)
                if opt_run.status == TrainingStatus.CANCELLED.value:
                    return

                fitness_values = []

                for particle in particles:
                    # Decode and evaluate
                    config_decoded = PSOService.decode_position(
                        particle["position"],
                        dataset.num_features,
                        dataset.num_classes,
                    )
                    fitness = PSOService.evaluate_fitness(
                        config_decoded, X_train, y_train, X_val, y_val
                    )
                    fitness_values.append(fitness)

                    # Update personal best
                    if fitness > particle["best_fitness"]:
                        particle["best_fitness"] = fitness
                        particle["best_position"] = particle["position"].copy()

                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle["position"].copy()

                # Update velocities and positions
                for particle in particles:
                    for key in particle["position"]:
                        r1, r2 = np.random.random(), np.random.random()

                        # Velocity update
                        cognitive = c1 * r1 * (particle["best_position"][key] - particle["position"][key])
                        social = c2 * r2 * (global_best_position[key] - particle["position"][key])
                        particle["velocity"][key] = w * particle["velocity"][key] + cognitive + social

                        # Position update
                        particle["position"][key] += particle["velocity"][key]

                        # Clamp to bounds
                        low, high = PSOService.BOUNDS[key]
                        particle["position"][key] = max(low, min(high, particle["position"][key]))

                # Record history
                history.append({
                    "iteration": iteration + 1,
                    "best_fitness": global_best_fitness,
                    "avg_fitness": sum(fitness_values) / len(fitness_values),
                })

                # Update database periodically
                if (iteration + 1) % 10 == 0:
                    opt_run.history = history
                    db.commit()

            # Final result
            best_config = PSOService.decode_position(
                global_best_position,
                dataset.num_features,
                dataset.num_classes,
            )

            opt_run.status = TrainingStatus.COMPLETED.value
            opt_run.best_result = {
                "network_config": best_config["network_config"],
                "accuracy": global_best_fitness,
                "loss": 1 - global_best_fitness,
            }
            opt_run.history = history
            db.commit()

        except Exception as e:
            logger.error(f"PSO optimization failed: {e}")
            opt_run.status = TrainingStatus.FAILED.value
            db.commit()
            raise e
        finally:
            db.close()