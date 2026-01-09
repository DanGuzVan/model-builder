import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from typing import Dict, Any, List, Tuple
import random

from app.config import settings
from app.database import SessionLocal
from app.models import OptimizationRun, Dataset, TrainingStatus
from app.services.neural_network import NeuralNetwork
from app.services.trainer import TrainerService


class Particle:
    """Represents a particle in PSO."""

    def __init__(self, bounds: Dict[str, Tuple[float, float]]):
        self.position = {}
        self.velocity = {}
        self.best_position = {}
        self.best_fitness = float("-inf")

        # Initialize position and velocity
        for key, (low, high) in bounds.items():
            self.position[key] = random.uniform(low, high)
            self.velocity[key] = random.uniform(-(high - low), high - low) * 0.1

        self.best_position = self.position.copy()


class PSOService:
    """Particle Swarm Optimization service for neural network architecture search."""

    # Search space bounds
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
                torch.from_numpy(X_train),
                torch.from_numpy(y_train),
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
                X_val_tensor = torch.from_numpy(X_val).to(device)
                y_val_tensor = torch.from_numpy(y_val).to(device)
                outputs = model(X_val_tensor)
                _, predicted = outputs.max(1)
                accuracy = predicted.eq(y_val_tensor).sum().item() / len(y_val)

            return accuracy

        except Exception:
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

            # Initialize particles
            particles = [Particle(PSOService.BOUNDS) for _ in range(num_particles)]
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
                        particle.position,
                        dataset.num_features,
                        dataset.num_classes,
                    )
                    fitness = PSOService.evaluate_fitness(
                        config_decoded, X_train, y_train, X_val, y_val
                    )
                    fitness_values.append(fitness)

                    # Update personal best
                    if fitness > particle.best_fitness:
                        particle.best_fitness = fitness
                        particle.best_position = particle.position.copy()

                    # Update global best
                    if fitness > global_best_fitness:
                        global_best_fitness = fitness
                        global_best_position = particle.position.copy()

                # Update velocities and positions
                for particle in particles:
                    for key in particle.position:
                        r1, r2 = random.random(), random.random()

                        # Velocity update
                        cognitive = c1 * r1 * (particle.best_position[key] - particle.position[key])
                        social = c2 * r2 * (global_best_position[key] - particle.position[key])
                        particle.velocity[key] = w * particle.velocity[key] + cognitive + social

                        # Position update
                        particle.position[key] += particle.velocity[key]

                        # Clamp to bounds
                        low, high = PSOService.BOUNDS[key]
                        particle.position[key] = max(low, min(high, particle.position[key]))

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
            opt_run.status = TrainingStatus.FAILED.value
            db.commit()
            raise e
        finally:
            db.close()
