"""
Training Router - API endpoints for neural network training.

Endpoints:
- POST /api/training - Start training (with background task)
- POST /api/training/start - Alias for starting training
- GET /api/training - List all training runs
- GET /api/training/{id} - Get training run details
- GET /api/training/{id}/metrics - Get metrics for plotting
- POST /api/training/{id}/cancel - Cancel a running training
- DELETE /api/training/{id} - Delete training run
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Optional

from app.database import get_db
from app.models import TrainingRun, Dataset, Network, TrainingStatus
from app.schemas import (
    TrainingRunCreate,
    TrainingRunResponse,
    TrainingRunList,
    TrainingMetrics,
)
from app.services.trainer import TrainerService

router = APIRouter()


@router.post("/", response_model=TrainingRunResponse)
async def create_training_run(
    training_run: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Create and start a new training run.
    
    Body: {dataset_id, network_id, learning_rate, batch_size, epochs}
    
    Runs training in background task.
    Returns training_run_id.
    """
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

    # Validate network input/output sizes match dataset
    network_config = network.config
    if network_config.get("input_size") != dataset.num_features:
        raise HTTPException(
            status_code=400,
            detail=f"Network input size ({network_config.get('input_size')}) "
                   f"doesn't match dataset features ({dataset.num_features})"
        )
    if network_config.get("output_size") != dataset.num_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Network output size ({network_config.get('output_size')}) "
                   f"doesn't match dataset classes ({dataset.num_classes})"
        )

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


@router.post("/start", response_model=TrainingRunResponse)
async def start_training(
    training_run: TrainingRunCreate,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    """
    Alias for POST /api/training - Start a new training run.
    
    Body: {dataset_id, network_id, learning_rate, batch_size, epochs, early_stopping_patience}
    """
    return await create_training_run(training_run, background_tasks, db)


@router.get("/", response_model=TrainingRunList)
async def list_training_runs(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    dataset_id: Optional[int] = Query(None, description="Filter by dataset ID"),
    network_id: Optional[int] = Query(None, description="Filter by network ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all training runs.
    
    Supports filtering by dataset_id, network_id, and status.
    """
    query = select(TrainingRun)
    
    if dataset_id is not None:
        query = query.where(TrainingRun.dataset_id == dataset_id)
    if network_id is not None:
        query = query.where(TrainingRun.network_id == network_id)
    if status is not None:
        query = query.where(TrainingRun.status == status)
    
    query = query.offset(skip).limit(limit).order_by(TrainingRun.created_at.desc())

    result = await db.execute(query)
    training_runs = result.scalars().all()

    # Count total (with same filters)
    count_query = select(TrainingRun)
    if dataset_id is not None:
        count_query = count_query.where(TrainingRun.dataset_id == dataset_id)
    if network_id is not None:
        count_query = count_query.where(TrainingRun.network_id == network_id)
    if status is not None:
        count_query = count_query.where(TrainingRun.status == status)
    
    count_result = await db.execute(count_query)
    total = len(count_result.scalars().all())

    return TrainingRunList(training_runs=training_runs, total=total)


@router.get("/{run_id}", response_model=TrainingRunResponse)
async def get_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific training run.
    
    Includes status, metrics history, best accuracy.
    """
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    return training_run


@router.get("/{run_id}/metrics")
async def get_training_metrics(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get metrics for plotting.
    
    Returns:
    - train_loss: List of training losses per epoch
    - val_loss: List of validation losses per epoch
    - train_accuracy: List of training accuracies per epoch (0-1 scale)
    - val_accuracy: List of validation accuracies per epoch (0-1 scale)
    - epoch_times: List of epoch durations in seconds
    - epochs_completed: Number of completed epochs
    - best_epoch: Epoch with best validation accuracy
    - best_accuracy: Best validation accuracy achieved
    """
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    metrics = training_run.metrics or {}
    
    # Calculate best epoch
    val_accuracy = metrics.get("val_accuracy", [])
    best_epoch = 0
    best_val_acc = 0.0
    
    if val_accuracy:
        best_val_acc = max(val_accuracy)
        best_epoch = val_accuracy.index(best_val_acc) + 1

    return {
        "run_id": run_id,
        "status": training_run.status,
        "train_loss": metrics.get("train_loss", []),
        "val_loss": metrics.get("val_loss", []),
        "train_accuracy": metrics.get("train_accuracy", []),
        "val_accuracy": metrics.get("val_accuracy", []),
        "epoch_times": metrics.get("epoch_times", []),
        "epochs_completed": len(metrics.get("train_loss", [])),
        "epochs_total": training_run.epochs,
        "best_epoch": best_epoch,
        "best_accuracy": training_run.best_accuracy or best_val_acc,
        "learning_rate": training_run.learning_rate,
        "batch_size": training_run.batch_size,
    }


@router.post("/{run_id}/cancel")
async def cancel_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Cancel a running training run.
    
    Only pending or running training runs can be cancelled.
    """
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    if training_run.status not in [TrainingStatus.PENDING.value, TrainingStatus.RUNNING.value]:
        raise HTTPException(
            status_code=400, 
            detail=f"Training run cannot be cancelled (current status: {training_run.status})"
        )

    training_run.status = TrainingStatus.CANCELLED.value
    await db.flush()

    return {
        "message": "Training run cancelled", 
        "id": run_id,
        "status": TrainingStatus.CANCELLED.value,
    }


@router.delete("/{run_id}")
async def delete_training_run(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a training run.
    
    Running training runs should be cancelled first.
    """
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Warn if still running
    if training_run.status == TrainingStatus.RUNNING.value:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete a running training run. Cancel it first."
        )

    await db.delete(training_run)

    return {"message": "Training run deleted successfully", "id": run_id}


@router.get("/{run_id}/summary")
async def get_training_summary(
    run_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a comprehensive summary of a training run.
    
    Includes training configuration, results, and related entities.
    """
    # Get training run
    result = await db.execute(select(TrainingRun).where(TrainingRun.id == run_id))
    training_run = result.scalar_one_or_none()

    if not training_run:
        raise HTTPException(status_code=404, detail="Training run not found")

    # Get related dataset
    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == training_run.dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()

    # Get related network
    network_result = await db.execute(
        select(Network).where(Network.id == training_run.network_id)
    )
    network = network_result.scalar_one_or_none()

    metrics = training_run.metrics or {}
    
    return {
        "training_run": {
            "id": training_run.id,
            "status": training_run.status,
            "learning_rate": training_run.learning_rate,
            "batch_size": training_run.batch_size,
            "epochs_configured": training_run.epochs,
            "epochs_completed": len(metrics.get("train_loss", [])),
            "best_accuracy": training_run.best_accuracy,
            "created_at": training_run.created_at.isoformat() if training_run.created_at else None,
        },
        "dataset": {
            "id": dataset.id if dataset else None,
            "name": dataset.name if dataset else "Unknown",
            "num_features": dataset.num_features if dataset else None,
            "num_classes": dataset.num_classes if dataset else None,
            "num_samples": dataset.num_samples if dataset else None,
        },
        "network": {
            "id": network.id if network else None,
            "name": network.name if network else "Unknown",
            "architecture": network.config.get("hidden_layers", []) if network else [],
            "dropout": network.config.get("dropout", 0) if network else 0,
            "parameter_count": network.config.get("parameter_count") if network else None,
        },
        "final_metrics": {
            "final_train_loss": metrics.get("train_loss", [None])[-1],
            "final_val_loss": metrics.get("val_loss", [None])[-1],
            "final_train_accuracy": metrics.get("train_accuracy", [None])[-1],
            "final_val_accuracy": metrics.get("val_accuracy", [None])[-1],
            "total_training_time": sum(metrics.get("epoch_times", [])),
        } if metrics else None,
    }