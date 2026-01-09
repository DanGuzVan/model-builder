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


class Individual:
    """Represents an individual in the genetic algorithm."""

    def __init__(self, genes: Dict[str, float] = None):
        if genes is None:
            genes = Individual.random_genes()
        self.genes = genes
        self.fitness = 0.0

    @staticmethod
    def random_genes() -> Dict[str, float]:
        """Generate random genes."""
        return {
            "num_layers": random.randint(1, 5),
            "layer_size": random.randint(16, 256),
            "dropout": random.uniform(0.0, 0.5),
            "learning_rate": random.uniform(-4, -1),  # log10 scale
        }

    def to_config(self, input_size: int, output_size: int) -> Dict[str, Any]:
        """Convert genes to network configuration."""
        num_layers = max(1, int(round(self.genes["num_layers"])))
        layer_size = max(16, int(round(self.genes["layer_size"])))
        dropout = max(0.0, min(0.5, self.genes["dropout"]))
        learning_rate = 10 ** self.genes["learning_rate"]

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


class GeneticService:
    """Genetic Algorithm service for neural network architecture search."""

    GENE_BOUNDS = {
        "num_layers": (1, 5),
        "layer_size": (16, 256),
        "dropout": (0.0, 0.5),
        "learning_rate": (-4, -1),
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
    def select_parents(population: List[Individual], num_parents: int) -> List[Individual]:
        """Tournament selection."""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        return parents

    @staticmethod
    def crossover(parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}

        for key in parent1.genes:
            if random.random() < 0.5:
                child1_genes[key] = parent1.genes[key]
                child2_genes[key] = parent2.genes[key]
            else:
                child1_genes[key] = parent2.genes[key]
                child2_genes[key] = parent1.genes[key]

        return Individual(child1_genes), Individual(child2_genes)

    @staticmethod
    def mutate(individual: Individual, mutation_rate: float):
        """Mutate an individual's genes."""
        for key in individual.genes:
            if random.random() < mutation_rate:
                low, high = GeneticService.GENE_BOUNDS[key]
                if isinstance(low, int):
                    individual.genes[key] = random.randint(int(low), int(high))
                else:
                    # Gaussian mutation
                    sigma = (high - low) * 0.1
                    individual.genes[key] += random.gauss(0, sigma)
                    individual.genes[key] = max(low, min(high, individual.genes[key]))

    @staticmethod
    def run_optimization(run_id: int):
        """Run GA optimization (background task)."""
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

            # GA parameters
            config = opt_run.config
            population_size = config.get("population_size", 50)
            generations = config.get("generations", 100)
            mutation_rate = config.get("mutation_rate", 0.1)
            crossover_rate = config.get("crossover_rate", 0.8)

            # Initialize population
            population = [Individual() for _ in range(population_size)]

            best_individual = None
            best_fitness = float("-inf")
            history = []

            # Evolution loop
            for generation in range(generations):
                # Check if cancelled
                db.refresh(opt_run)
                if opt_run.status == TrainingStatus.CANCELLED.value:
                    return

                # Evaluate fitness
                for individual in population:
                    config_decoded = individual.to_config(
                        dataset.num_features,
                        dataset.num_classes,
                    )
                    individual.fitness = GeneticService.evaluate_fitness(
                        config_decoded, X_train, y_train, X_val, y_val
                    )

                    if individual.fitness > best_fitness:
                        best_fitness = individual.fitness
                        best_individual = Individual(individual.genes.copy())
                        best_individual.fitness = individual.fitness

                # Record history
                avg_fitness = sum(ind.fitness for ind in population) / len(population)
                history.append({
                    "iteration": generation + 1,
                    "best_fitness": best_fitness,
                    "avg_fitness": avg_fitness,
                })

                # Selection
                parents = GeneticService.select_parents(population, population_size)

                # Create next generation
                next_population = []

                # Elitism: keep best individual
                if best_individual:
                    next_population.append(Individual(best_individual.genes.copy()))

                while len(next_population) < population_size:
                    parent1, parent2 = random.sample(parents, 2)

                    if random.random() < crossover_rate:
                        child1, child2 = GeneticService.crossover(parent1, parent2)
                    else:
                        child1 = Individual(parent1.genes.copy())
                        child2 = Individual(parent2.genes.copy())

                    GeneticService.mutate(child1, mutation_rate)
                    GeneticService.mutate(child2, mutation_rate)

                    next_population.append(child1)
                    if len(next_population) < population_size:
                        next_population.append(child2)

                population = next_population

                # Update database periodically
                if (generation + 1) % 10 == 0:
                    opt_run.history = history
                    db.commit()

            # Final result
            best_config = best_individual.to_config(
                dataset.num_features,
                dataset.num_classes,
            )

            opt_run.status = TrainingStatus.COMPLETED.value
            opt_run.best_result = {
                "network_config": best_config["network_config"],
                "accuracy": best_fitness,
                "loss": 1 - best_fitness,
            }
            opt_run.history = history
            db.commit()

        except Exception as e:
            opt_run.status = TrainingStatus.FAILED.value
            db.commit()
            raise e
        finally:
            db.close()
