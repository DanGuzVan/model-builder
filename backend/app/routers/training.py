from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import TrainingRun, Dataset, Network, TrainingStatus
from app.schemas import (
    TrainingRunCreate,
    TrainingRunResponse,
    TrainingRunList,
)
from app.services.trainer import TrainerService

router = APIRouter()


@router.post("/", response_model=TrainingRunResponse)
async def create_training_run(
    training_run: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """Create and start a new training run."""
    # Verify dataset exists
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == training_run.dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Verify network exists
    network_result = await db.execute(
        select(Network).where(Network.id == training_run.network_id)
    )
    network = network_result.scalar_one_or_none()
    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    # Create training run record
    db_training_run = TrainingRun(
        dataset_id=training_run.dataset_id,
        network_id=training_run.network_id,
        learning_rate=training_run.learning_rate,
        batch_size=training_run.batch_size,
        epochs=training_run.epochs,
        status=TrainingStatus.PENDING.value,
    )
    db.add(db_training_run)
    await db.flush()
    await db.refresh(db_training_run)

    # Start training in background
    background_tasks.add_task(
        TrainerService.run_training,
        db_training_run.id,
    )

    return db_training_run


@router.get("/", response_model=TrainingRunList)
async def list_training_runs(
    skip: int = 0,
    limit: int = 100,
    dataset_id: int = None,
    db: AsyncSession = Depends(get_db),
):
    """List all training runs."""
    query = select(TrainingRun)
    if dataset_id:
        query = query.where(TrainingRun.dataset_id == dataset_id)
    query = query.offset(skip).limit(limit).order_by(TrainingRun.created_at.desc())

    result = await db.execute(query)
    training_runs = result.scalars().all()

    count_query = select(TrainingRun)
    if dataset_id:
        count_query = count_query.where(TrainingRun.dataset_id == dataset_id)
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return TrainingRunList(training_runs=training_runs, total=total)


@router.get("/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific training run."""
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return training_run


@router.post("/{run_id}/cancel")
async def cancel_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Cancel a running training run."""
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if training_run.status not in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
        raise HTTPException(status_code=400, detail="Training run cannot be cancelled")

    training_run.status = TrainingStatus.CANCELLED.value
    await db.flush()

    return {"message": "Training run cancelled"}


@router.delete("/{run_id}")
async def delete_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a training run."""
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    await db.delete(training_run)

    return {"message": "Training run deleted successfully"}
