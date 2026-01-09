from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List

from app.database import get_db
from app.models import Network
from app.schemas import (
    NetworkCreate,
    NetworkUpdate,
    NetworkResponse,
    NetworkList,
    NetworkSuggestionRequest,
    NetworkSuggestionResponse,
)
from app.services.llm_service import LLMService

router = APIRouter()


@router.post("/", response_model=NetworkResponse)
async def create_network(
    network: NetworkCreate,
    db: AsyncSession = Depends(get_db),
):
    """Create a new neural network configuration."""
    db_network = Network(
        name=network.name,
        config=network.config.model_dump(),
    )
    db.add(db_network)
    await db.flush()
    await db.refresh(db_network)

    return db_network


@router.get("/", response_model=NetworkList)
async def list_networks(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all network configurations."""
    result = await db.execute(
        select(Network).offset(skip).limit(limit).order_by(Network.created_at.desc())
    )
    networks = result.scalars().all()

    count_result = await db.execute(select(Network))
    total = len(count_result.scalars().all())

    return NetworkList(networks=networks, total=total)


@router.get("/{network_id}", response_model=NetworkResponse)
async def get_network(
    network_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific network configuration."""
    result = await db.execute(select(Network).where(Network.id == network_id))
    network = result.scalar_one_or_none()

    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    return network


@router.put("/{network_id}", response_model=NetworkResponse)
async def update_network(
    network_id: int,
    network_update: NetworkUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a network configuration."""
    result = await db.execute(select(Network).where(Network.id == network_id))
    network = result.scalar_one_or_none()

    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    if network_update.name is not None:
        network.name = network_update.name
    if network_update.config is not None:
        network.config = network_update.config.model_dump()

    await db.flush()
    await db.refresh(network)

    return network


@router.delete("/{network_id}")
async def delete_network(
    network_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a network configuration."""
    result = await db.execute(select(Network).where(Network.id == network_id))
    network = result.scalar_one_or_none()

    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    await db.delete(network)

    return {"message": "Network deleted successfully"}


@router.post("/suggest", response_model=NetworkSuggestionResponse)
async def suggest_network(
    request: NetworkSuggestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """Use LLM to suggest a network architecture for a dataset."""
    from app.models import Dataset

    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    llm_service = LLMService()
    suggestion = await llm_service.suggest_network_architecture(
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
        num_samples=dataset.num_samples,
        task_description=request.task_description,
    )

    return suggestion
