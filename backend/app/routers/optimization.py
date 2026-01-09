"""
Optimization Router - API endpoints for neural network optimization using PSO and GA.

Endpoints:
- POST /api/optimization - Start optimization (with background task)
- POST /api/optimization/start - Alias for starting optimization
- GET /api/optimization - List all optimization runs
- GET /api/optimization/{id} - Get optimization run details
- GET /api/optimization/{id}/history - Get optimization history for plotting
- POST /api/optimization/{id}/cancel - Cancel a running optimization
- DELETE /api/optimization/{id} - Delete optimization run
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_db
from app.models import OptimizationRun, Dataset, TrainingStatus, OptimizationAlgorithm
from app.schemas import (
    OptimizationRunCreate,
    OptimizationRunResponse,
    OptimizationRunList,
    PSOConfig,
    GAConfig,
)
from app.services.pso import PSOService
from app.services.genetic import GeneticService

router = APIRouter()


def validate_pso_config(config: dict) -> dict:
    """Validate and normalize PSO configuration."""
    default_pso = PSOConfig()
    return {
        "num_particles": config.get("num_particles", default_pso.num_particles),
        "max_iterations": config.get("max_iterations", default_pso.max_iterations),
        "w": config.get("w", default_pso.w),
        "c1": config.get("c1", default_pso.c1),
        "c2": config.get("c2", default_pso.c2),
    }


def validate_ga_config(config: dict) -> dict:
    """Validate and normalize GA configuration."""
    default_ga = GAConfig()
    return {
        "population_size": config.get("population_size", default_ga.population_size),
        "generations": config.get("generations", default_ga.generations),
        "mutation_rate": config.get("mutation_rate", default_ga.mutation_rate),
        "crossover_rate": config.get("crossover_rate", default_ga.crossover_rate),
    }


@router.post("/", response_model=OptimizationRunResponse)
async def create_optimization_run(
    optimization_run: OptimizationRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Create and start a new optimization run.
    
    Body: {dataset_id, algorithm: 'pso'|'ga', config: {population_size, iterations, ...}}
    
    PSO Config:
    - num_particles: Number of particles (default 30)
    - max_iterations: Maximum iterations (default 100)
    - w: Inertia weight (default 0.7)
    - c1: Cognitive coefficient (default 1.5)
    - c2: Social coefficient (default 1.5)
    
    GA Config:
    - population_size: Size of population (default 50)
    - generations: Number of generations (default 100)
    - mutation_rate: Probability of mutation (default 0.1)
    - crossover_rate: Probability of crossover (default 0.8)
    
    Returns optimization_run_id.
    """
    # Verify dataset exists
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == optimization_run.dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Validate algorithm
    algorithm = optimization_run.algorithm
    if algorithm not in [OptimizationAlgorithm.PSO, OptimizationAlgorithm.GA]:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid algorithm: {algorithm}. Must be 'pso' or 'ga'"
        )

    # Validate and normalize config based on algorithm
    config = optimization_run.config or {}
    if algorithm == OptimizationAlgorithm.PSO:
        validated_config = validate_pso_config(config)
    else:
        validated_config = validate_ga_config(config)

    # Create optimization run record
    db_optimization_run = OptimizationRun(
        dataset_id=optimization_run.dataset_id,
        algorithm=algorithm.value,
        config=validated_config,
        status=TrainingStatus.PENDING.value,
    )
    db.add(db_optimization_run)
    await db.flush()
    await db.refresh(db_optimization_run)

    # Start optimization in background
    if algorithm == OptimizationAlgorithm.PSO:
        background_tasks.add_task(
            PSOService.run_optimization,
            db_optimization_run.id,
        )
    else:
        background_tasks.add_task(
            GeneticService.run_optimization,
            db_optimization_run.id,
        )

    return db_optimization_run


@router.post("/start", response_model=OptimizationRunResponse)
async def start_optimization(
    optimization_run: OptimizationRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Alias for POST /api/optimization - Start a new optimization run.
    
    Body: {dataset_id, algorithm: 'pso'|'ga', config: {...}}
    """
    return await create_optimization_run(optimization_run, background_tasks, db)


@router.get("/", response_model=OptimizationRunList)
async def list_optimization_runs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    algorithm: Optional[str] = Query(None, description="Filter by algorithm (pso/ga)"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all optimization runs.
    
    Supports filtering by dataset_id, algorithm, and status.
    """
    query = select(OptimizationRun)
    
    if dataset_id is not None:
        query = query.where(OptimizationRun.dataset_id == dataset_id)
    if algorithm is not None:
        query = query.where(OptimizationRun.algorithm == algorithm)
    if status is not None:
        query = query.where(OptimizationRun.status == status)
    
    query = query.offset(skip).limit(limit).order_by(OptimizationRun.created_at.desc())

    result = await db.execute(query)
    optimization_runs = result.scalars().all()

    # Count total (with same filters)
    count_query = select(OptimizationRun)
    if dataset_id is not None:
        count_query = count_query.where(OptimizationRun.dataset_id == dataset_id)
    if algorithm is not None:
        count_query = count_query.where(OptimizationRun.algorithm == algorithm)
    if status is not None:
        count_query = count_query.where(OptimizationRun.status == status)
    
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return OptimizationRunList(optimization_runs=optimization_runs, total=total)


@router.get("/{run_id}", response_model=OptimizationRunResponse)
async def get_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific optimization run.
    
    Includes status, best result, history.
    """
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    return optimization_run


@router.get("/{run_id}/history")
async def get_optimization_history(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get optimization history for plotting.
    
    Returns:
    - iterations: List of iteration numbers
    - best_fitness: List of best fitness values per iteration
    - avg_fitness: List of average fitness values per iteration
    - algorithm: The algorithm used
    - config: The configuration used
    - best_result: The best result found
    """
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    history = optimization_run.history or []
    
    return {
        "run_id": run_id,
        "status": optimization_run.status,
        "algorithm": optimization_run.algorithm,
        "config": optimization_run.config,
        "iterations": [h.get("iteration", i+1) for i, h in enumerate(history)],
        "best_fitness": [h.get("best_fitness", 0) for h in history],
        "avg_fitness": [h.get("avg_fitness", 0) for h in history],
        "total_iterations": len(history),
        "best_result": optimization_run.best_result,
        "improvement": (
            history[-1].get("best_fitness", 0) - history[0].get("best_fitness", 0)
            if len(history) >= 2 else 0
        ),
    }


@router.post("/{run_id}/cancel")
async def cancel_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running optimization run.
    
    Only pending or running optimization runs can be cancelled.
    """
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    if optimization_run.status not in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
        raise HTTPException(
            status_code=400, 
            detail=f"Optimization run cannot be cancelled (current status: {optimization_run.status})"
        )

    optimization_run.status = TrainingStatus.CANCELLED.value
    await db.flush()

    return {
        "message": "Optimization run cancelled", 
        "id": run_id,
        "status": TrainingStatus.CANCELLED.value,
    }


@router.delete("/{run_id}")
async def delete_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete an optimization run.
    
    Running optimization runs should be cancelled first.
    """
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    # Warn if still running
    if optimization_run.status == TrainingStatus.RUNNING.value:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running optimization run. Cancel it first."
        )

    await db.delete(optimization_run)

    return {"message": "Optimization run deleted successfully", "id": run_id}


@router.get("/{run_id}/summary")
async def get_optimization_summary(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a comprehensive summary of an optimization run.
    
    Includes configuration, results, and related dataset.
    """
    # Get optimization run
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    # Get related dataset
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == optimization_run.dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()

    history = optimization_run.history or []
    best_result = optimization_run.best_result or {}
    
    return {
        "optimization_run": {
            "id": optimization_run.id,
            "status": optimization_run.status,
            "algorithm": optimization_run.algorithm,
            "config": optimization_run.config,
            "iterations_completed": len(history),
            "best_accuracy": best_result.get("accuracy"),
            "created_at": optimization_run.created_at.isoformat() if optimization_run.created_at else None,
        },
        "dataset": {
            "id": dataset.id if dataset else None,
            "name": dataset.name if dataset else "Unknown",
            "num_features": dataset.num_features if dataset else None,
            "num_classes": dataset.num_classes if dataset else None,
            "num_samples": dataset.num_samples if dataset else None,
        },
        "best_network": best_result.get("network_config") if best_result else None,
        "convergence": {
            "initial_fitness": history[0].get("best_fitness") if history else None,
            "final_fitness": history[-1].get("best_fitness") if history else None,
            "improvement": (
                history[-1].get("best_fitness", 0) - history[0].get("best_fitness", 0)
                if len(history) >= 2 else 0
            ),
            "converged_at_iteration": next(
                (i for i in range(len(history) - 1) 
                 if abs(history[i+1].get("best_fitness", 0) - history[i].get("best_fitness", 0)) < 0.0001),
                None
            ) if history else None,
        },
    }


@router.get("/algorithms/info")
async def get_algorithms_info():
    """
    Get information about available optimization algorithms.
    
    Returns descriptions, default configurations, and typical use cases.
    """
    return {
        "algorithms": [
            {
                "id": "pso",
                "name": "Particle Swarm Optimization",
                "description": "A population-based optimization algorithm inspired by social behavior of bird flocking. Best for continuous hyperparameter optimization like learning rate and dropout.",
                "default_config": PSOConfig().model_dump(),
                "config_schema": {
                    "num_particles": {"type": "integer", "min": 5, "max": 200, "description": "Number of particles in the swarm"},
                    "max_iterations": {"type": "integer", "min": 10, "max": 1000, "description": "Maximum number of iterations"},
                    "w": {"type": "float", "min": 0, "max": 1, "description": "Inertia weight - controls momentum"},
                    "c1": {"type": "float", "min": 0, "max": 3, "description": "Cognitive coefficient - attraction to personal best"},
                    "c2": {"type": "float", "min": 0, "max": 3, "description": "Social coefficient - attraction to global best"},
                },
                "use_cases": ["Learning rate optimization", "Dropout rate tuning", "Layer size optimization"],
                "scientific_reference": "Kennedy & Eberhart (1995)",
            },
            {
                "id": "ga",
                "name": "Genetic Algorithm",
                "description": "An evolutionary algorithm that mimics natural selection. Best for discrete architecture decisions like number of layers and network topology.",
                "default_config": GAConfig().model_dump(),
                "config_schema": {
                    "population_size": {"type": "integer", "min": 10, "max": 500, "description": "Size of the population"},
                    "generations": {"type": "integer", "min": 10, "max": 1000, "description": "Number of generations"},
                    "mutation_rate": {"type": "float", "min": 0, "max": 1, "description": "Probability of gene mutation"},
                    "crossover_rate": {"type": "float", "min": 0, "max": 1, "description": "Probability of crossover"},
                },
                "use_cases": ["Architecture search", "Layer count optimization", "Network topology design"],
                "scientific_reference": "Holland (1975)",
            },
        ],
    }