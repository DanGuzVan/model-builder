"""
Datasets Router - API endpoints for dataset management.

Endpoints:
- POST /api/datasets/upload - Upload CSV file
- GET /api/datasets - List all datasets
- GET /api/datasets/{id} - Get dataset details with preview
- GET /api/datasets/{id}/preview - Get dataset preview
- DELETE /api/datasets/{id} - Delete dataset and file
"""

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional
import pandas as pd
import os
import uuid

from app.database import get_db
from app.config import settings
from app.models import Dataset
from app.schemas import DatasetResponse, DatasetList

router = APIRouter()


@router.post("/upload", response_model=DatasetResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a CSV dataset file.
    
    - Accept multipart file upload
    - Save to uploads/ folder
    - Parse to get num_features, num_classes, num_samples
    - Store metadata in DB
    - Return dataset info
    """
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    if not name.strip():
        raise HTTPException(status_code=400, detail="Dataset name is required")

    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.upload_dir, unique_filename)

    # Read file content
    content = await file.read()
    
    # Check file size
    if len(content) > settings.max_upload_size:
        raise HTTPException(
            status_code=400, 
            detail=f"File too large. Maximum size is {settings.max_upload_size / (1024*1024):.0f}MB"
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

    # Save file
    try:
        os.makedirs(settings.upload_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    # Parse CSV to get metadata
    try:
        df = pd.read_csv(file_path)
        
        if df.empty:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        if len(df.columns) < 2:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail="CSV must have at least 2 columns (features + target)"
            )
        
        num_samples = len(df)
        num_features = len(df.columns) - 1  # Assuming last column is target
        num_classes = df.iloc[:, -1].nunique()
        
        if num_classes < 2:
            os.remove(file_path)
            raise HTTPException(
                status_code=400, 
                detail="Target column must have at least 2 unique classes"
            )
            
    except pd.errors.EmptyDataError:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="CSV file is empty or malformed")
    except pd.errors.ParserError as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error processing CSV: {str(e)}")

    # Create database record
    dataset = Dataset(
        name=name.strip(),
        filename=unique_filename,
        num_features=num_features,
        num_classes=num_classes,
        num_samples=num_samples,
    )
    db.add(dataset)
    await db.flush()
    await db.refresh(dataset)

    return dataset


@router.get("/", response_model=DatasetList)
async def list_datasets(
    skip: int = Query(0, ge=0, description="Number of records to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    List all datasets.
    
    Returns list with id, name, num_features, num_classes, num_samples.
    Supports pagination via skip and limit parameters.
    """
    result = await db.execute(
        select(Dataset).offset(skip).limit(limit).order_by(Dataset.created_at.desc())
    )
    datasets = result.scalars().all()

    count_result = await db.execute(select(Dataset))
    total = len(count_result.scalars().all())

    return DatasetList(datasets=datasets, total=total)


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Get a specific dataset by ID.
    
    Returns dataset metadata including feature count, class count, and sample count.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    rows: int = Query(10, ge=1, le=100, description="Number of rows to preview"),
    db: AsyncSession = Depends(get_db),
):
    """
    Preview first N rows of a dataset.
    
    Returns:
    - columns: List of column names
    - data: List of row data
    - dtypes: Dictionary of column data types
    - statistics: Basic statistics for numerical columns
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(settings.upload_dir, dataset.filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset file not found on disk")
    
    try:
        # Read full dataset for statistics, but only return preview rows
        df_full = pd.read_csv(file_path)
        df_preview = df_full.head(rows)
        
        # Get basic statistics for numerical columns
        numeric_cols = df_full.select_dtypes(include=['number']).columns.tolist()
        statistics = {}
        for col in numeric_cols:
            statistics[col] = {
                "min": float(df_full[col].min()),
                "max": float(df_full[col].max()),
                "mean": float(df_full[col].mean()),
                "std": float(df_full[col].std()) if len(df_full) > 1 else 0.0,
            }
        
        return {
            "columns": df_preview.columns.tolist(),
            "data": df_preview.values.tolist(),
            "dtypes": df_preview.dtypes.astype(str).to_dict(),
            "statistics": statistics,
            "total_rows": len(df_full),
            "preview_rows": len(df_preview),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Delete a dataset and its associated file.
    
    This will also remove the CSV file from the uploads directory.
    """
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete file from disk
    file_path = os.path.join(settings.upload_dir, dataset.filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except Exception as e:
            # Log warning but continue with database deletion
            print(f"Warning: Could not delete file {file_path}: {e}")

    await db.delete(dataset)

    return {"message": "Dataset deleted successfully", "id": dataset_id}