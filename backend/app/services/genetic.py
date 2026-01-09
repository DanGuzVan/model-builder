"""
Genetic Algorithm for Neural Network Architecture Optimization.

This module implements a Genetic Algorithm (GA) for optimizing
neural network architectures, including discrete decisions like
number of layers and layer sizes.

Scientific basis:
- GA excels at discrete/combinatorial optimization
- Tournament selection balances exploration and exploitation
- Elitism preserves best solutions across generations
- Adaptive mutation prevents premature convergence

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
import random

from app.config import settings
from app.database import SessionLocal
from app.models import OptimizationRun, Dataset, TrainingStatus
from app.services.neural_network import NeuralNetwork, ConfigurableClassifier
from app.services.trainer import TrainerService

logger = logging.getLogger(__name__)


@dataclass
class Individual:
    """
    Represents an individual in the genetic population.
    
    Genes encode the neural network architecture:
    - genes[0]: num_layers (1-5)
    - genes[1]: layer1_size (16-256)
    - genes[2]: layer2_size (16-256)
    - genes[3]: layer3_size (16-256)
    - genes[4]: learning_rate (log10 scale: -4 to -1)
    - genes[5]: dropout (0.0 to 0.5)
    """
    genes: np.ndarray
    fitness: float = 0.0
    generation: int = 0
    
    def copy(self) -> "Individual":
        """Create a deep copy of this individual."""
        return Individual(
            genes=self.genes.copy(),
            fitness=self.fitness,
            generation=self.generation
        )


@dataclass
class GeneDefinition:
    """Defines properties of a single gene."""
    name: str
    min_value: float
    max_value: float
    is_integer: bool = False
    is_log_scale: bool = False


class GeneticOptimizer:
    """
    Genetic Algorithm for neural network architecture optimization.
    
    Uses evolutionary principles to search the architecture space:
    - Selection: Tournament selection (k=3)
    - Crossover: Single-point crossover
    - Mutation: Gaussian mutation with adaptive rate
    - Elitism: Preserve top N individuals
    
    Gene Encoding (6 genes):
    [num_layers, layer1_size, layer2_size, layer3_size, learning_rate, dropout]
    
    Args:
        population_size: Number of individuals (default 15)
        n_generations: Maximum generations (default 10)
        crossover_rate: Probability of crossover (default 0.8)
        mutation_rate: Probability of gene mutation (default 0.2)
        elitism: Number of top individuals to preserve (default 2)
        tournament_size: Size of tournament for selection (default 3)
        eval_epochs: Epochs for fitness evaluation (default 10)
    
    Example:
        >>> optimizer = GeneticOptimizer(population_size=20, n_generations=15)
        >>> result = optimizer.optimize(X, y, input_size=10, output_size=3)
        >>> print(result['best_config'])
    """
    
    # Gene definitions
    GENE_DEFINITIONS = [
        GeneDefinition("num_layers", 1, 5, is_integer=True),
        GeneDefinition("layer1_size", 16, 256, is_integer=True),
        GeneDefinition("layer2_size", 16, 256, is_integer=True),
        GeneDefinition("layer3_size", 16, 256, is_integer=True),
        GeneDefinition("learning_rate", -4, -1, is_log_scale=True),
        GeneDefinition("dropout", 0.0, 0.5),
    ]
    
    def __init__(
        self,
        population_size: int = 15,
        n_generations: int = 10,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism: int = 2,
        tournament_size: int = 3,
        eval_epochs: int = 10
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.tournament_size = tournament_size
        self.eval_epochs = eval_epochs
        
        # State
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict[str, Any]] = []
        
        # Device for training
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _random_genes(self) -> np.ndarray:
        """Generate random genes within valid bounds."""
        genes = np.zeros(len(self.GENE_DEFINITIONS))
        for i, gene_def in enumerate(self.GENE_DEFINITIONS):
            if gene_def.is_integer:
                genes[i] = random.randint(int(gene_def.min_value), int(gene_def.max_value))
            else:
                genes[i] = random.uniform(gene_def.min_value, gene_def.max_value)
        return genes
    
    def _clip_genes(self, genes: np.ndarray) -> np.ndarray:
        """Clip genes to valid bounds."""
        for i, gene_def in enumerate(self.GENE_DEFINITIONS):
            genes[i] = np.clip(genes[i], gene_def.min_value, gene_def.max_value)
            if gene_def.is_integer:
                genes[i] = round(genes[i])
        return genes
    
    def _initialize_population(self) -> List[Individual]:
        """
        Create random initial population.
        
        Returns:
            List of randomly initialized individuals
        """
        population = []
        for _ in range(self.population_size):
            genes = self._random_genes()
            individual = Individual(genes=genes, fitness=0.0, generation=0)
            population.append(individual)
        return population
    
    def _decode_genes(self, genes: np.ndarray, input_size: int, output_size: int) -> Dict[str, Any]:
        """
        Convert genes to architecture configuration.
        
        Args:
            genes: Gene array
            input_size: Number of input features
            output_size: Number of output classes
            
        Returns:
            Network configuration dict
        """
        num_layers = max(1, int(round(genes[0])))
        layer_sizes = [
            max(16, int(round(genes[1]))),
            max(16, int(round(genes[2]))),
            max(16, int(round(genes[3]))),
        ]
        learning_rate = 10 ** genes[4]  # Convert from log scale
        dropout = max(0.0, min(0.5, genes[5]))
        
        # Build hidden layers based on num_layers
        hidden_layers = layer_sizes[:num_layers]
        
        return {
            "network_config": {
                "input_size": input_size,
                "hidden_layers": hidden_layers,
                "output_size": output_size,
                "dropout": dropout,
            },
            "learning_rate": learning_rate,
            "num_layers": num_layers,
        }
    
    def _evaluate(
        self,
        config: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> float:
        """
        Train network with config and return validation accuracy.
        
        Args:
            config: Decoded network configuration
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Validation accuracy (0.0 to 1.0), or 0.0 on error
        """
        try:
            model = ConfigurableClassifier.from_config(config["network_config"]).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
            
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
            logger.warning(f"GA evaluation failed: {e}")
            return 0.0
    
    def _tournament_select(self, population: List[Individual]) -> Individual:
        """
        Select individual via tournament selection.
        
        Args:
            population: Current population
            
        Returns:
            Selected individual (copy)
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        winner = max(tournament, key=lambda x: x.fitness)
        return winner.copy()
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """
        Single-point crossover, return two children.
        
        Args:
            parent1, parent2: Parent individuals
            
        Returns:
            Tuple of two child individuals
        """
        if random.random() > self.crossover_rate:
            # No crossover, return copies of parents
            return parent1.copy(), parent2.copy()
        
        # Single-point crossover
        point = random.randint(1, len(parent1.genes) - 1)
        
        child1_genes = np.concatenate([parent1.genes[:point], parent2.genes[point:]])
        child2_genes = np.concatenate([parent2.genes[:point], parent1.genes[point:]])
        
        return (
            Individual(genes=child1_genes, fitness=0.0),
            Individual(genes=child2_genes, fitness=0.0)
        )
    
    def _mutate(self, individual: Individual):
        """
        Mutate genes with mutation_rate probability.
        
        Uses Gaussian mutation for continuous genes and
        uniform random for discrete genes.
        
        Args:
            individual: Individual to mutate (modified in place)
        """
        for i, gene_def in enumerate(self.GENE_DEFINITIONS):
            if random.random() < self.mutation_rate:
                if gene_def.is_integer:
                    # Uniform random mutation for integers
                    individual.genes[i] = random.randint(
                        int(gene_def.min_value),
                        int(gene_def.max_value)
                    )
                else:
                    # Gaussian mutation for continuous values
                    sigma = (gene_def.max_value - gene_def.min_value) * 0.1
                    individual.genes[i] += random.gauss(0, sigma)
        
        # Ensure genes are within bounds
        individual.genes = self._clip_genes(individual.genes)
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        input_size: int,
        output_size: int,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Run genetic algorithm optimization.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            input_size: Number of input features
            output_size: Number of output classes
            callback: Optional callback called each generation
            
        Returns:
            Dict containing:
            - best_config: Best network configuration found
            - best_fitness: Best validation accuracy
            - history: List of dicts with generation stats
        """
        # Split data for evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Initialize population
        self.population = self._initialize_population()
        self.best_individual = None
        self.history = []
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Evaluate fitness of all individuals
            for individual in self.population:
                config = self._decode_genes(individual.genes, input_size, output_size)
                individual.fitness = self._evaluate(
                    config, X_train, y_train, X_val, y_val
                )
                individual.generation = generation
                
                # Track best individual
                if self.best_individual is None or individual.fitness > self.best_individual.fitness:
                    self.best_individual = individual.copy()
            
            # Calculate statistics
            fitness_values = [ind.fitness for ind in self.population]
            generation_stats = {
                "iteration": generation + 1,
                "best_fitness": self.best_individual.fitness,
                "avg_fitness": np.mean(fitness_values),
                "std_fitness": np.std(fitness_values),
                "min_fitness": np.min(fitness_values),
                "max_fitness": np.max(fitness_values),
                "progress": (generation + 1) / self.n_generations
            }
            self.history.append(generation_stats)
            
            # Call callback if provided
            if callback:
                callback(generation_stats)
            
            logger.info(
                f"GA Generation {generation + 1}/{self.n_generations}: "
                f"Best={self.best_individual.fitness:.4f}, "
                f"Avg={np.mean(fitness_values):.4f}"
            )
            
            # Create next generation
            next_population = []
            
            # Elitism: Keep top N individuals
            sorted_population = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for i in range(min(self.elitism, len(sorted_population))):
                elite = sorted_population[i].copy()
                elite.generation = generation + 1
                next_population.append(elite)
            
            # Fill rest with offspring
            while len(next_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(self.population)
                parent2 = self._tournament_select(self.population)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                self._mutate(child1)
                self._mutate(child2)
                
                # Add to next generation
                child1.generation = generation + 1
                child2.generation = generation + 1
                next_population.append(child1)
                if len(next_population) < self.population_size:
                    next_population.append(child2)
            
            self.population = next_population
        
        # Final evaluation of last generation
        for individual in self.population:
            config = self._decode_genes(individual.genes, input_size, output_size)
            individual.fitness = self._evaluate(
                config, X_train, y_train, X_val, y_val
            )
            if individual.fitness > self.best_individual.fitness:
                self.best_individual = individual.copy()
        
        # Return results
        best_config = self._decode_genes(self.best_individual.genes, input_size, output_size)
        return {
            "best_config": best_config["network_config"],
            "best_fitness": self.best_individual.fitness,
            "best_learning_rate": best_config["learning_rate"],
            "history": self.history,
        }


class GeneticService:
    """
    Service class for GA optimization (integrates with FastAPI).
    
    This maintains backward compatibility with the existing API.
    """

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
    def select_parents(population: List[Dict], num_parents: int) -> List[Dict]:
        """Tournament selection."""
        parents = []
        for _ in range(num_parents):
            tournament = random.sample(population, min(3, len(population)))
            winner = max(tournament, key=lambda x: x["fitness"])
            parents.append(winner)
        return parents

    @staticmethod
    def crossover(parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Uniform crossover."""
        child1_genes = {}
        child2_genes = {}

        for key in parent1["genes"]:
            if random.random() < 0.5:
                child1_genes[key] = parent1["genes"][key]
                child2_genes[key] = parent2["genes"][key]
            else:
                child1_genes[key] = parent2["genes"][key]
                child2_genes[key] = parent1["genes"][key]

        return (
            {"genes": child1_genes, "fitness": 0.0},
            {"genes": child2_genes, "fitness": 0.0}
        )

    @staticmethod
    def mutate(individual: Dict, mutation_rate: float):
        """Mutate an individual's genes."""
        for key in individual["genes"]:
            if random.random() < mutation_rate:
                low, high = GeneticService.GENE_BOUNDS[key]
                if isinstance(low, int):
                    individual["genes"][key] = random.randint(int(low), int(high))
                else:
                    # Gaussian mutation
                    sigma = (high - low) * 0.1
                    individual["genes"][key] += random.gauss(0, sigma)
                    individual["genes"][key] = max(low, min(high, individual["genes"][key]))

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
            population = []
            for _ in range(population_size):
                genes = {}
                for key, (low, high) in GeneticService.GENE_BOUNDS.items():
                    if isinstance(low, int):
                        genes[key] = random.randint(low, high)
                    else:
                        genes[key] = random.uniform(low, high)
                population.append({"genes": genes, "fitness": 0.0})

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
                    # Decode genes to config
                    genes = individual["genes"]
                    num_layers = max(1, int(round(genes["num_layers"])))
                    layer_size = max(16, int(round(genes["layer_size"])))
                    dropout = max(0.0, min(0.5, genes["dropout"]))
                    learning_rate = 10 ** genes["learning_rate"]

                    config_decoded = {
                        "network_config": {
                            "input_size": dataset.num_features,
                            "hidden_layers": [layer_size] * num_layers,
                            "output_size": dataset.num_classes,
                            "dropout": dropout,
                        },
                        "learning_rate": learning_rate,
                    }

                    individual["fitness"] = GeneticService.evaluate_fitness(
                        config_decoded, X_train, y_train, X_val, y_val
                    )

                    if individual["fitness"] > best_fitness:
                        best_fitness = individual["fitness"]
                        best_individual = {
                            "genes": individual["genes"].copy(),
                            "fitness": individual["fitness"]
                        }

                # Record history
                avg_fitness = sum(ind["fitness"] for ind in population) / len(population)
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
                    next_population.append({
                        "genes": best_individual["genes"].copy(),
                        "fitness": 0.0
                    })

                while len(next_population) < population_size:
                    parent1, parent2 = random.sample(parents, 2)

                    if random.random() < crossover_rate:
                        child1, child2 = GeneticService.crossover(parent1, parent2)
                    else:
                        child1 = {"genes": parent1["genes"].copy(), "fitness": 0.0}
                        child2 = {"genes": parent2["genes"].copy(), "fitness": 0.0}

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
            best_genes = best_individual["genes"]
            num_layers = max(1, int(round(best_genes["num_layers"])))
            layer_size = max(16, int(round(best_genes["layer_size"])))
            dropout = max(0.0, min(0.5, best_genes["dropout"]))

            best_config = {
                "input_size": dataset.num_features,
                "hidden_layers": [layer_size] * num_layers,
                "output_size": dataset.num_classes,
                "dropout": dropout,
            }

            opt_run.status = TrainingStatus.COMPLETED.value
            opt_run.best_result = {
                "network_config": best_config,
                "accuracy": best_fitness,
                "loss": 1 - best_fitness,
            }
            opt_run.history = history
            db.commit()

        except Exception as e:
            logger.error(f"GA optimization failed: {e}")
            opt_run.status = TrainingStatus.FAILED.value
            db.commit()
            raise e
        finally:
            db.close()