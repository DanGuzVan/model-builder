from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


# Enums
class TrainingStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizationAlgorithm(str, Enum):
    PSO = "pso"
    GA = "ga"


# Dataset Schemas
class DatasetBase(BaseModel):
    name: str


class DatasetCreate(DatasetBase):
    pass


class DatasetResponse(DatasetBase):
    id: int
    filename: str
    num_features: int
    num_classes: int
    num_samples: int
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetList(BaseModel):
    datasets: List[DatasetResponse]
    total: int


# Network Config Schema
class NetworkConfig(BaseModel):
    input_size: int = Field(..., ge=1)
    hidden_layers: List[int] = Field(..., min_length=1)
    output_size: int = Field(..., ge=1)
    dropout: float = Field(default=0.0, ge=0.0, le=1.0)


# Network Schemas
class NetworkBase(BaseModel):
    name: str
    config: NetworkConfig


class NetworkCreate(NetworkBase):
    pass


class NetworkUpdate(BaseModel):
    name: Optional[str] = None
    config: Optional[NetworkConfig] = None


class NetworkResponse(NetworkBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class NetworkList(BaseModel):
    networks: List[NetworkResponse]
    total: int


# Training Metrics Schema
class TrainingMetrics(BaseModel):
    train_loss: List[float] = []
    val_loss: List[float] = []
    train_accuracy: List[float] = []
    val_accuracy: List[float] = []
    epoch_times: List[float] = []


# Training Run Schemas
class TrainingRunBase(BaseModel):
    dataset_id: int
    network_id: int
    learning_rate: float = Field(default=0.001, gt=0, le=1)
    batch_size: int = Field(default=32, ge=1, le=1024)
    epochs: int = Field(default=100, ge=1, le=10000)


class TrainingRunCreate(TrainingRunBase):
    pass


class TrainingRunResponse(TrainingRunBase):
    id: int
    status: TrainingStatus
    best_accuracy: Optional[float] = None
    metrics: Optional[TrainingMetrics] = None
    created_at: datetime

    class Config:
        from_attributes = True


class TrainingRunList(BaseModel):
    training_runs: List[TrainingRunResponse]
    total: int


# PSO Config Schema
class PSOConfig(BaseModel):
    num_particles: int = Field(default=30, ge=5, le=200)
    max_iterations: int = Field(default=100, ge=10, le=1000)
    w: float = Field(default=0.7, ge=0, le=1, description="Inertia weight")
    c1: float = Field(default=1.5, ge=0, le=3, description="Cognitive coefficient")
    c2: float = Field(default=1.5, ge=0, le=3, description="Social coefficient")


# Genetic Algorithm Config Schema
class GAConfig(BaseModel):
    population_size: int = Field(default=50, ge=10, le=500)
    generations: int = Field(default=100, ge=10, le=1000)
    mutation_rate: float = Field(default=0.1, ge=0, le=1)
    crossover_rate: float = Field(default=0.8, ge=0, le=1)


# Optimization Result Schema
class OptimizationResult(BaseModel):
    network_config: NetworkConfig
    accuracy: float
    loss: float


# Optimization History Entry Schema
class OptimizationHistoryEntry(BaseModel):
    iteration: int
    best_fitness: float
    avg_fitness: float


# Optimization Run Schemas
class OptimizationRunBase(BaseModel):
    dataset_id: int
    algorithm: OptimizationAlgorithm


class OptimizationRunCreatePSO(OptimizationRunBase):
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.PSO
    config: PSOConfig = PSOConfig()


class OptimizationRunCreateGA(OptimizationRunBase):
    algorithm: OptimizationAlgorithm = OptimizationAlgorithm.GA
    config: GAConfig = GAConfig()


class OptimizationRunCreate(BaseModel):
    dataset_id: int
    algorithm: OptimizationAlgorithm
    config: Dict[str, Any]  # Either PSOConfig or GAConfig as dict


class OptimizationRunResponse(BaseModel):
    id: int
    dataset_id: int
    algorithm: OptimizationAlgorithm
    status: TrainingStatus
    config: Dict[str, Any]
    best_result: Optional[OptimizationResult] = None
    history: Optional[List[OptimizationHistoryEntry]] = None
    created_at: datetime

    class Config:
        from_attributes = True


class OptimizationRunList(BaseModel):
    optimization_runs: List[OptimizationRunResponse]
    total: int


# LLM Service Schemas
class LLMPromptRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = None


class LLMResponse(BaseModel):
    response: str
    model: str


class NetworkSuggestionRequest(BaseModel):
    dataset_id: int
    task_description: Optional[str] = None


class NetworkSuggestionResponse(BaseModel):
    suggested_config: NetworkConfig
    reasoning: str


# Health Check Schema
class HealthCheck(BaseModel):
    status: str
    database: str
    ollama: str
