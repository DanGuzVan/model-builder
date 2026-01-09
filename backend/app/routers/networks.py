"""
Networks Router - API endpoints for neural network architecture management.

Endpoints:
- POST /api/networks - Create network architecture
- GET /api/networks - List all networks
- GET /api/networks/{id} - Get network details with parameter count
- PUT /api/networks/{id} - Update network configuration
- DELETE /api/networks/{id} - Delete network
- POST /api/networks/suggest - Get LLM-based architecture suggestion
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional

from app.database import get_db
from app.models import Network, Dataset
from app.schemas import (
    NetworkCreate,
    NetworkUpdate,
    NetworkResponse,
    NetworkList,
    NetworkSuggestionRequest,
    NetworkSuggestionResponse,
    NetworkConfig,
)
from app.services.llm_service import LLMService

router = APIRouter()


def calculate_parameter_count(config: dict) -> int:
    """
    Calculate total number of parameters in the network.
    
    For a fully connected network:
    - Input to first hidden: input_size * hidden[0] + hidden[0] (bias)
    - Hidden to hidden: hidden[i] * hidden[i+1] + hidden[i+1] (bias)
    - Last hidden to output: hidden[-1] * output_size + output_size (bias)
    """
    input_size = config["input_size"]
    hidden_layers = config["hidden_layers"]
    output_size = config["output_size"]
    
    total_params = 0
    prev_size = input_size
    
    for hidden_size in hidden_layers:
        # Weights + biases
        total_params += prev_size * hidden_size + hidden_size
        prev_size = hidden_size
    
    # Output layer
    total_params += prev_size * output_size + output_size
    
    return total_params


def validate_network_config(config: NetworkConfig) -> None:
    """Validate network configuration."""
    if config.input_size <= 0:
        raise HTTPException(
            status_code=400, 
            detail="input_size must be positive"
        )
    
    if config.output_size <= 0:
        raise HTTPException(
            status_code=400, 
            detail="output_size must be positive"
        )
    
    if not config.hidden_layers or len(config.hidden_layers) == 0:
        raise HTTPException(
            status_code=400, 
            detail="At least one hidden layer is required"
        )
    
    for i, layer_size in enumerate(config.hidden_layers):
        if layer_size <= 0:
            raise HTTPException(
                status_code=400, 
                detail=f"Hidden layer {i+1} size must be positive"
            )
        if layer_size > 4096:
            raise HTTPException(
                status_code=400, 
                detail=f"Hidden layer {i+1} size exceeds maximum (4096)"
            )
    
    if len(config.hidden_layers) > 20:
        raise HTTPException(
            status_code=400, 
            detail="Maximum 20 hidden layers allowed"
        )
    
    if config.dropout < 0 or config.dropout >= 1:
        raise HTTPException(
            status_code=400, 
            detail="Dropout must be between 0 and 1 (exclusive)"
        )


@router.post("/", response_model=NetworkResponse)
async def create_network(
    network: NetworkCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new neural network configuration.
    
    Body: {name, config: {input_size, hidden_layers: [int], output_size, dropout}}
    
    Validates architecture and stores in DB.
    Returns network info with parameter count.
    """
    # Validate the configuration
    validate_network_config(network.config)
    
    if not network.name or not network.name.strip():
        raise HTTPException(status_code=400, detail="Network name is required")
    
    # Calculate parameter count
    config_dict = network.config.model_dump()
    param_count = calculate_parameter_count(config_dict)
    config_dict["parameter_count"] = param_count
    
    db_network = Network(
        name=network.name.strip(),
        config=config_dict,
    )
    db.add(db_network)
    await db.flush()
    await db.refresh(db_network)

    return db_network


@router.get("/", response_model=NetworkList)
async def list_networks(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all network configurations.
    
    Supports pagination via skip and limit parameters.
    """
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
    """
    Get a specific network configuration.
    
    Returns network details including configuration and parameter count.
    """
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
    """
    Update a network configuration.
    
    Can update name and/or configuration.
    """
    result = await db.execute(select(Network).where(Network.id == network_id))
    network = result.scalar_one_or_none()

    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    if network_update.name is not None:
        if not network_update.name.strip():
            raise HTTPException(status_code=400, detail="Network name cannot be empty")
        network.name = network_update.name.strip()
    
    if network_update.config is not None:
        # Validate the new configuration
        validate_network_config(network_update.config)
        
        config_dict = network_update.config.model_dump()
        param_count = calculate_parameter_count(config_dict)
        config_dict["parameter_count"] = param_count
        network.config = config_dict

    await db.flush()
    await db.refresh(network)

    return network


@router.delete("/{network_id}")
async def delete_network(
    network_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a network configuration.
    
    Note: This will fail if there are training runs associated with this network.
    """
    result = await db.execute(select(Network).where(Network.id == network_id))
    network = result.scalar_one_or_none()

    if not network:
        raise HTTPException(status_code=404, detail="Network not found")

    await db.delete(network)

    return {"message": "Network deleted successfully", "id": network_id}


@router.post("/suggest", response_model=NetworkSuggestionResponse)
async def suggest_network(
    request: NetworkSuggestionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Use LLM to suggest a network architecture for a dataset.
    
    The LLM analyzes the dataset characteristics and suggests:
    - Appropriate hidden layer sizes
    - Dropout rate
    - Reasoning for the choices
    
    Falls back to heuristic-based suggestions if LLM is unavailable.
    """
    # Get dataset
    result = await db.execute(select(Dataset).where(Dataset.id == request.dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        llm_service = LLMService()
        suggestion = await llm_service.suggest_network_architecture(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            num_samples=dataset.num_samples,
            task_description=request.task_description,
        )
        return suggestion
    except Exception as e:
        # If LLM fails, provide heuristic suggestion with error context
        llm_service = LLMService()
        suggestion = llm_service._heuristic_suggestion(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            num_samples=dataset.num_samples,
        )
        suggestion.reasoning = f"(Heuristic fallback - LLM unavailable) {suggestion.reasoning}"
        return suggestion


@router.post("/validate")
async def validate_network_architecture(
    config: NetworkConfig,
):
    """
    Validate a network architecture without creating it.
    
    Returns validation status and parameter count.
    """
    try:
        validate_network_config(config)
        config_dict = config.model_dump()
        param_count = calculate_parameter_count(config_dict)
        
        return {
            "valid": True,
            "parameter_count": param_count,
            "config": config_dict,
            "architecture_summary": f"{config.input_size} -> {' -> '.join(map(str, config.hidden_layers))} -> {config.output_size}",
        }
    except HTTPException as e:
        return {
            "valid": False,
            "error": e.detail,
            "parameter_count": None,
        }