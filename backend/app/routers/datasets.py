from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
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
    """Upload a CSV dataset file."""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")

    # Generate unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.upload_dir, unique_filename)

    # Save file
    content = await file.read()
    if len(content) > settings.max_upload_size:
        raise HTTPException(status_code=400, detail="File too large")

    with open(file_path, "wb") as f:
        f.write(content)

    # Parse CSV to get metadata
    try:
        df = pd.read_csv(file_path)
        num_samples = len(df)
        num_features = len(df.columns) - 1  # Assuming last column is target
        num_classes = df.iloc[:, -1].nunique()
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Error parsing CSV: {str(e)}")

    # Create database record
    dataset = Dataset(
        name=name,
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
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all datasets."""
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
    """Get a specific dataset by ID."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Delete a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Delete file
    file_path = os.path.join(settings.upload_dir, dataset.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    await db.delete(dataset)

    return {"message": "Dataset deleted successfully"}


@router.get("/{dataset_id}/preview")
async def preview_dataset(
    dataset_id: int,
    rows: int = 10,
    db: AsyncSession = Depends(get_db),
):
    """Preview first N rows of a dataset."""
    result = await db.execute(select(Dataset).where(Dataset.id == dataset_id))
    dataset = result.scalar_one_or_none()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    file_path = os.path.join(settings.upload_dir, dataset.filename)
    df = pd.read_csv(file_path, nrows=rows)

    return {
        "columns": df.columns.tolist(),
        "data": df.values.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }
