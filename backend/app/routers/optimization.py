from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import OptimizationRun, Dataset, TrainingStatus, OptimizationAlgorithm
from app.schemas import (
    OptimizationRunCreate,
    OptimizationRunResponse,
    OptimizationRunList,
)
from app.services.pso import PSOService
from app.services.genetic import GeneticService

router = APIRouter()


@router.post("/", response_model=OptimizationRunResponse)
async def create_optimization_run(
    optimization_run: OptimizationRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create and start a new optimization run."""
    # Verify dataset exists
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == optimization_run.dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Create optimization run record
    db_optimization_run = OptimizationRun(
        dataset_id=optimization_run.dataset_id,
        algorithm=optimization_run.algorithm.value,
        config=optimization_run.config,
        status=TrainingStatus.PENDING.value,
    )
    db.add(db_optimization_run)
    await db.flush()
    await db.refresh(db_optimization_run)

    # Start optimization in background
    if optimization_run.algorithm == OptimizationAlgorithm.PSO:
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


@router.get("/", response_model=OptimizationRunList)
async def list_optimization_runs(
    skip: int = 0,
    limit: int = 100,
    dataset_id: int = None,
    algorithm: str = None,
    db: AsyncSession = Depends(get_db),
):
    """List all optimization runs."""
    query = select(OptimizationRun)
    if dataset_id:
        query = query.where(OptimizationRun.dataset_id == dataset_id)
    if algorithm:
        query = query.where(OptimizationRun.algorithm == algorithm)
    query = query.offset(skip).limit(limit).order_by(OptimizationRun.created_at.desc())

    result = await db.execute(query)
    optimization_runs = result.scalars().all()

    count_query = select(OptimizationRun)
    if dataset_id:
        count_query = count_query.where(OptimizationRun.dataset_id == dataset_id)
    if algorithm:
        count_query = count_query.where(OptimizationRun.algorithm == algorithm)
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return OptimizationRunList(optimization_runs=optimization_runs, total=total)


@router.get("/{run_id}", response_model=OptimizationRunResponse)
async def get_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific optimization run."""
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    return optimization_run


@router.post("/{run_id}/cancel")
async def cancel_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running optimization run."""
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    if optimization_run.status not in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
        raise HTTPException(status_code=400, detail="Optimization run cannot be cancelled")

    optimization_run.status = TrainingStatus.CANCELLED.value
    await db.flush()

    return {"message": "Optimization run cancelled"}


@router.delete("/{run_id}")
async def delete_optimization_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete an optimization run."""
    result = await db.execute(
        select(OptimizationRun).where(OptimizationRun.id == run_id)
    )
    optimization_run = result.scalar_one_or_none()

    if not optimization_run:
        raise HTTPException(status_code=404, detail="Optimization run not found")

    await db.delete(optimization_run)

    return {"message": "Optimization run deleted successfully"}
