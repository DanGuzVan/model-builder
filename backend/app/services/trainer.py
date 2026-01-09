import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time
from typing import Dict, Any, Tuple

from app.config import settings
from app.database import SessionLocal
from app.models import TrainingRun, Dataset, Network, TrainingStatus
from app.services.neural_network import NeuralNetwork


class TrainerService:
    """Service for training neural networks."""

    @staticmethod
    def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a dataset from file."""
        file_path = os.path.join(settings.upload_dir, filename)
        df = pd.read_csv(file_path)

        # Assume last column is target
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Encode labels if string
        if y.dtype == object:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        return X.astype(np.float32), y.astype(np.int64)

    @staticmethod
    def create_data_loaders(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        test_size: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train),
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val),
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    @staticmethod
    def run_training(run_id: int):
        """Run training for a specific training run (background task)."""
        db = SessionLocal()
        try:
            # Get training run
            training_run = db.query(TrainingRun).filter(TrainingRun.id == run_id).first()
            if not training_run:
                return

            # Update status to running
            training_run.status = TrainingStatus.RUNNING.value
            db.commit()

            # Get dataset and network
            dataset = db.query(Dataset).filter(Dataset.id == training_run.dataset_id).first()
            network_config = db.query(Network).filter(Network.id == training_run.network_id).first()

            if not dataset or not network_config:
                training_run.status = TrainingStatus.FAILED.value
                db.commit()
                return

            # Load data
            X, y = TrainerService.load_dataset(dataset.filename)
            train_loader, val_loader = TrainerService.create_data_loaders(
                X, y, training_run.batch_size
            )

            # Create model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = NeuralNetwork.from_config(network_config.config).to(device)

            # Setup training
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=training_run.learning_rate)

            # Training metrics
            metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
                "epoch_times": [],
            }
            best_accuracy = 0.0

            # Training loop
            for epoch in range(training_run.epochs):
                # Check if cancelled
                db.refresh(training_run)
                if training_run.status == TrainingStatus.CANCELLED.value:
                    return

                epoch_start = time.time()

                # Training phase
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()

                # Validation phase
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                # Calculate metrics
                train_accuracy = train_correct / train_total
                val_accuracy = val_correct / val_total
                epoch_time = time.time() - epoch_start

                metrics["train_loss"].append(train_loss / len(train_loader))
                metrics["val_loss"].append(val_loss / len(val_loader))
                metrics["train_accuracy"].append(train_accuracy)
                metrics["val_accuracy"].append(val_accuracy)
                metrics["epoch_times"].append(epoch_time)

                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy

                # Update database every 10 epochs
                if (epoch + 1) % 10 == 0 or epoch == training_run.epochs - 1:
                    training_run.metrics = metrics
                    training_run.best_accuracy = best_accuracy
                    db.commit()

            # Training completed
            training_run.status = TrainingStatus.COMPLETED.value
            training_run.metrics = metrics
            training_run.best_accuracy = best_accuracy
            db.commit()

        except Exception as e:
            training_run.status = TrainingStatus.FAILED.value
            db.commit()
            raise e
        finally:
            db.close()
