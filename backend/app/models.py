from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, JSON, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum

from app.database import Base


class TrainingStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationAlgorithm(str, enum.Enum):
    PSO = "pso"
    GA = "ga"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    num_features = Column(Integer, nullable=False)
    num_classes = Column(Integer, nullable=False)
    num_samples = Column(Integer, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_runs = relationship("TrainingRun", back_populates="dataset")
    optimization_runs = relationship("OptimizationRun", back_populates="dataset")


class Network(Base):
    __tablename__ = "networks"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    config = Column(JSON, nullable=False)
    # config structure: {
    #     "input_size": int,
    #     "hidden_layers": [int, ...],
    #     "output_size": int,
    #     "dropout": float
    # }
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    training_runs = relationship("TrainingRun", back_populates="network")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    network_id = Column(Integer, ForeignKey("networks.id"), nullable=False)
    status = Column(String(50), default=TrainingStatus.PENDING.value)
    learning_rate = Column(Float, nullable=False)
    batch_size = Column(Integer, nullable=False)
    epochs = Column(Integer, nullable=False)
    best_accuracy = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)
    # metrics structure: {
    #     "train_loss": [float, ...],
    #     "val_loss": [float, ...],
    #     "train_accuracy": [float, ...],
    #     "val_accuracy": [float, ...],
    #     "epoch_times": [float, ...]
    # }
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dataset = relationship("Dataset", back_populates="training_runs")
    network = relationship("Network", back_populates="training_runs")


class OptimizationRun(Base):
    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    algorithm = Column(String(50), nullable=False)  # 'pso' or 'ga'
    status = Column(String(50), default=TrainingStatus.PENDING.value)
    config = Column(JSON, nullable=False)
    # PSO config: {
    #     "num_particles": int,
    #     "max_iterations": int,
    #     "w": float (inertia),
    #     "c1": float (cognitive),
    #     "c2": float (social)
    # }
    # GA config: {
    #     "population_size": int,
    #     "generations": int,
    #     "mutation_rate": float,
    #     "crossover_rate": float
    # }
    best_result = Column(JSON, nullable=True)
    # best_result: {
    #     "network_config": {...},
    #     "accuracy": float,
    #     "loss": float
    # }
    history = Column(JSON, nullable=True)
    # history: [
    #     {"iteration": int, "best_fitness": float, "avg_fitness": float},
    #     ...
    # ]
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    dataset = relationship("Dataset", back_populates="optimization_runs")
