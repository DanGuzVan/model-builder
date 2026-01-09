"""
Training Service with Monitoring and Early Stopping.

This module provides comprehensive training functionality including:
- Data loading and preprocessing (DataProcessor)
- Early stopping to prevent overfitting (EarlyStopping)
- Training engine with full monitoring (Trainer)

Scientific basis:
- Prechelt (1998): Early stopping is highly effective for regularization
- Learning rate is the most critical hyperparameter (Bengio, 2012)
- Monitoring train/validation loss detects overfitting

Author: NN Optimizer Project
"""

import os
import time
import copy
from typing import Dict, Any, Tuple, Optional, Callable, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from app.config import settings
from app.database import SessionLocal
from app.models import TrainingRun, Dataset, Network, TrainingStatus
from app.services.neural_network import NeuralNetwork, ConfigurableClassifier


class DataProcessor:
    """
    Handle data loading and preprocessing for neural network training.
    
    Features:
    - CSV loading with automatic target detection
    - Feature normalization using StandardScaler
    - Label encoding for categorical targets
    - Stratified train/validation splitting
    - DataLoader creation with configurable batch size
    
    Scientific basis:
    - Feature normalization speeds up training and improves convergence
    - Stratified splitting ensures class balance in train/val sets
    
    Example:
        >>> processor = DataProcessor()
        >>> X, y = processor.load_csv('data.csv')
        >>> X_scaled = processor.preprocess(X, fit=True)
        >>> train_loader, val_loader = processor.get_dataloaders(X_scaled, y, batch_size=32)
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self._is_fitted = False
        self._label_encoder_fitted = False
    
    def load_csv(
        self, 
        filepath: str,
        target_column: Optional[Union[str, int]] = None,
        drop_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load CSV file and separate features from target.
        
        Args:
            filepath: Path to CSV file
            target_column: Column name or index for target. If None, uses last column.
            drop_columns: List of column names to drop (e.g., ID columns)
        
        Returns:
            Tuple of (features array, target array)
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is empty or malformed
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        df = pd.read_csv(filepath)
        
        if df.empty:
            raise ValueError("CSV file is empty")
        
        # Drop specified columns
        if drop_columns:
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
        
        # Determine target column
        if target_column is None:
            # Assume last column is target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
        elif isinstance(target_column, str):
            y = df[target_column].values
            X = df.drop(columns=[target_column]).values
        else:
            y = df.iloc[:, target_column].values
            X = df.drop(df.columns[target_column], axis=1).values
        
        # Encode labels if categorical
        if y.dtype == object or not np.issubdtype(y.dtype, np.number):
            y = self.label_encoder.fit_transform(y)
            self._label_encoder_fitted = True
        else:
            y = y.astype(np.int64)
        
        return X.astype(np.float32), y.astype(np.int64)
    
    def preprocess(
        self, 
        X: np.ndarray, 
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Scientific basis:
        - Normalization ensures all features contribute equally
        - Speeds up gradient descent convergence
        - Prevents features with large ranges from dominating
        
        Args:
            X: Feature array of shape (n_samples, n_features)
            fit: If True, fit the scaler on this data. If False, use existing fit.
        
        Returns:
            Normalized feature array
        
        Raises:
            ValueError: If fit=False and scaler hasn't been fitted yet
        """
        if fit:
            self._is_fitted = True
            return self.scaler.fit_transform(X).astype(np.float32)
        else:
            if not self._is_fitted:
                raise ValueError("Scaler has not been fitted. Call preprocess with fit=True first.")
            return self.scaler.transform(X).astype(np.float32)
    
    def get_dataloaders(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        val_split: float = 0.2,
        random_state: int = 42,
        shuffle_train: bool = True,
        num_workers: int = 0
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create train and validation DataLoaders with stratified splitting.
        
        Scientific basis:
        - Stratified split ensures class distribution is preserved
        - Separate validation set enables overfitting detection
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Target array (n_samples,)
            batch_size: Number of samples per batch
            val_split: Fraction of data for validation (default 0.2 = 20%)
            random_state: Random seed for reproducibility
            shuffle_train: Whether to shuffle training data each epoch
            num_workers: Number of worker processes for data loading
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=val_split,
            random_state=random_state,
            stratify=y
        )
        
        # Create TensorDatasets
        train_dataset = TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train)
        )
        val_dataset = TensorDataset(
            torch.from_numpy(X_val),
            torch.from_numpy(y_val)
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        return train_loader, val_loader
    
    def get_class_names(self) -> Optional[np.ndarray]:
        """Get original class names if labels were encoded."""
        if self._label_encoder_fitted:
            return self.label_encoder.classes_
        return None
    
    def get_num_classes(self, y: np.ndarray) -> int:
        """Get number of unique classes."""
        return len(np.unique(y))
    
    def get_feature_stats(self) -> Optional[Dict[str, np.ndarray]]:
        """Get feature statistics from fitted scaler."""
        if not self._is_fitted:
            return None
        return {
            "mean": self.scaler.mean_,
            "std": self.scaler.scale_,
            "var": self.scaler.var_
        }


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    
    Scientific basis:
    Prechelt (1998) showed early stopping is one of the most effective 
    regularization techniques. It prevents overfitting by:
    1. Monitoring validation loss (or other metric)
    2. Stopping when no improvement for 'patience' epochs
    3. Restoring best model weights
    
    Args:
        patience: Number of epochs to wait for improvement (default 10)
        min_delta: Minimum change to qualify as improvement (default 0.001)
        mode: 'min' for loss (lower is better), 'max' for accuracy
        restore_best: Whether to restore best weights when stopping
        verbose: Whether to print early stopping messages
    
    Example:
        >>> early_stopper = EarlyStopping(patience=10, min_delta=0.001)
        >>> for epoch in range(epochs):
        ...     train_model()
        ...     val_loss = validate()
        ...     if early_stopper.check(val_loss, model):
        ...         print("Early stopping triggered!")
        ...         early_stopper.restore_best(model)
        ...         break
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min',
        restore_best: bool = True,
        verbose: bool = False
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.verbose = verbose
        
        # Internal state
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        
        # Set comparison function based on mode
        if mode == 'min':
            self.is_better = lambda a, b: a < b - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda a, b: a > b + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def check(self, score: float, model: nn.Module, epoch: int = 0) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation metric (loss or accuracy)
            model: Model to save weights from
            epoch: Current epoch number (for logging)
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            # Save best weights (deep copy to avoid reference issues)
            self.best_weights = copy.deepcopy(model.state_dict())
            if self.verbose:
                print(f"EarlyStopping: New best score {score:.6f} at epoch {epoch}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
        
        return False
    
    def restore_best(self, model: nn.Module) -> bool:
        """
        Restore best weights to model.
        
        Args:
            model: Model to restore weights to
        
        Returns:
            True if weights were restored, False if no best weights saved
        """
        if self.best_weights is not None and self.restore_best:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"EarlyStopping: Restored best weights from epoch {self.best_epoch}")
            return True
        return False
    
    def reset(self):
        """Reset early stopping state for new training run."""
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        if self.mode == 'min':
            self.best_score = float('inf')
        else:
            self.best_score = float('-inf')
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state for logging/debugging."""
        return {
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "patience": self.patience,
            "stopped_epoch": self.stopped_epoch
        }


class Trainer:
    """
    Training engine with comprehensive monitoring and early stopping.
    
    Features:
    - Training and validation loops with metrics tracking
    - Early stopping support
    - Learning rate scheduling
    - Callback support for custom logging
    - Complete training history
    
    Scientific basis:
    - Adam optimizer (Kingma & Ba, 2014) for adaptive learning rates
    - CrossEntropyLoss for multi-class classification
    - Monitoring both loss and accuracy for complete picture
    
    Args:
        model: PyTorch model to train
        learning_rate: Initial learning rate (most critical hyperparameter)
        device: Device to train on ('cpu', 'cuda', or torch.device)
        optimizer: Optimizer class (default: Adam)
        optimizer_kwargs: Additional optimizer arguments
    
    Example:
        >>> model = ConfigurableClassifier(10, [64, 32], 3)
        >>> trainer = Trainer(model, learning_rate=0.001)
        >>> history = trainer.fit(train_loader, val_loader, epochs=100,
        ...                       early_stopping=EarlyStopping(patience=10))
    """
    
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        device: Union[str, torch.device] = 'cpu',
        optimizer: type = optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None
    ):
        # Handle device
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device
        
        # Move model to device
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        
        # Setup optimizer
        optimizer_kwargs = optimizer_kwargs or {}
        self.optimizer = optimizer(
            self.model.parameters(),
            lr=learning_rate,
            **optimizer_kwargs
        )
        
        # Loss function (CrossEntropyLoss includes softmax)
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'epoch_times': [],
            'learning_rates': []
        }
        
        # Scheduler (optional)
        self.scheduler = None
    
    def set_scheduler(
        self,
        scheduler_class,
        **scheduler_kwargs
    ):
        """
        Set learning rate scheduler.
        
        Args:
            scheduler_class: PyTorch scheduler class
            scheduler_kwargs: Arguments for scheduler
        """
        self.scheduler = scheduler_class(self.optimizer, **scheduler_kwargs)
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Move to device
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = (correct / total) * 100
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Validate model on validation set.
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (average_loss, accuracy_percentage)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()
        
        avg_loss = total_loss / total
        accuracy = (correct / total) * 100
        
        return avg_loss, accuracy
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: Optional[EarlyStopping] = None,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Full training loop with monitoring.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Maximum number of epochs to train
            early_stopping: Optional EarlyStopping instance
            callback: Optional function called each epoch with metrics dict
            verbose: Whether to print progress
        
        Returns:
            Training history dictionary with all metrics
        
        The callback receives a dict with keys:
        - epoch: Current epoch number
        - train_loss, val_loss: Loss values
        - train_acc, val_acc: Accuracy percentages
        - epoch_time: Time taken for this epoch
        - learning_rate: Current learning rate
        """
        # Reset history for new training run
        for key in self.history:
            self.history[key] = []
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Timing
            epoch_time = time.time() - epoch_start
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            self.history['learning_rates'].append(current_lr)
            
            # Track best accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # Prepare metrics dict for callback
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'epoch_time': epoch_time,
                'learning_rate': current_lr,
                'best_val_acc': best_val_acc
            }
            
            # Call callback if provided
            if callback:
                callback(metrics)
            
            # Print progress
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                      f"Time: {epoch_time:.2f}s")
            
            # Learning rate scheduler step
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Early stopping check
            if early_stopping:
                if early_stopping.check(val_loss, self.model, epoch + 1):
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                        print(f"Best validation loss: {early_stopping.best_score:.4f} "
                              f"at epoch {early_stopping.best_epoch}")
                    early_stopping.restore_best(self.model)
                    break
        
        return self.history
    
    def get_history(self) -> Dict[str, List]:
        """Return training metrics history."""
        return self.history.copy()
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            test_loader: DataLoader for test data
        
        Returns:
            Dict with 'loss' and 'accuracy' keys
        """
        loss, acc = self.validate(test_loader)
        return {'loss': loss, 'accuracy': acc}
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'learning_rate': self.learning_rate
        }
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])


class TrainerService:
    """
    Service class for training neural networks (integrates with FastAPI).
    
    This class provides the interface expected by the API routers.
    """

    @staticmethod
    def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess a dataset from file."""
        file_path = os.path.join(settings.upload_dir, filename)
        processor = DataProcessor()
        X, y = processor.load_csv(file_path)
        X = processor.preprocess(X, fit=True)
        return X, y

    @staticmethod
    def create_data_loaders(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        test_size: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation data loaders."""
        processor = DataProcessor()
        return processor.get_dataloaders(X, y, batch_size, val_split=test_size)

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

            # Setup trainer with early stopping
            trainer = Trainer(model, learning_rate=training_run.learning_rate, device=device)
            early_stopping = EarlyStopping(patience=15, min_delta=0.001, verbose=False)

            # Training metrics storage
            metrics = {
                "train_loss": [],
                "val_loss": [],
                "train_accuracy": [],
                "val_accuracy": [],
                "epoch_times": [],
            }
            best_accuracy = 0.0

            # Custom callback for database updates
            def training_callback(epoch_metrics: Dict[str, Any]):
                nonlocal best_accuracy, metrics
                
                metrics["train_loss"].append(epoch_metrics['train_loss'])
                metrics["val_loss"].append(epoch_metrics['val_loss'])
                metrics["train_accuracy"].append(epoch_metrics['train_acc'] / 100)  # Convert to 0-1
                metrics["val_accuracy"].append(epoch_metrics['val_acc'] / 100)
                metrics["epoch_times"].append(epoch_metrics['epoch_time'])
                
                if epoch_metrics['val_acc'] / 100 > best_accuracy:
                    best_accuracy = epoch_metrics['val_acc'] / 100
                
                # Check if cancelled
                db.refresh(training_run)
                if training_run.status == TrainingStatus.CANCELLED.value:
                    raise InterruptedError("Training cancelled")
                
                # Update database every 10 epochs
                if epoch_metrics['epoch'] % 10 == 0 or epoch_metrics['epoch'] == training_run.epochs:
                    training_run.metrics = metrics
                    training_run.best_accuracy = best_accuracy
                    db.commit()

            # Run training
            try:
                trainer.fit(
                    train_loader, 
                    val_loader, 
                    epochs=training_run.epochs,
                    early_stopping=early_stopping,
                    callback=training_callback,
                    verbose=False
                )
            except InterruptedError:
                return

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