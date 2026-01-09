"""
LLM Router - API endpoints for LLM-powered analysis and suggestions.

Endpoints:
- POST /api/llm/analyze-training - Analyze training results
- POST /api/llm/suggest-architecture - Suggest network architecture
- POST /api/llm/explain-optimization - Explain optimization results
- GET /api/llm/status - Check LLM service status
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import logging

from app.services.llm_service import get_llm_service, LLMService

logger = logging.getLogger(__name__)
router = APIRouter()


# Request/Response Models

class TrainingMetricsInput(BaseModel):
    """Input model for training analysis."""
    train_loss: List[float] = Field(default=[], description="Training losses per epoch")
    val_loss: List[float] = Field(default=[], description="Validation losses per epoch")
    train_accuracy: List[float] = Field(default=[], description="Training accuracies (0-1 scale)")
    val_accuracy: List[float] = Field(default=[], description="Validation accuracies (0-1 scale)")


class TrainingConfigInput(BaseModel):
    """Input model for training configuration."""
    learning_rate: float = Field(default=0.001, description="Learning rate used")
    batch_size: int = Field(default=32, description="Batch size used")
    epochs: int = Field(default=100, description="Total epochs configured")


class AnalyzeTrainingRequest(BaseModel):
    """Request body for training analysis."""
    metrics: TrainingMetricsInput
    config: TrainingConfigInput


class AnalyzeTrainingResponse(BaseModel):
    """Response for training analysis."""
    analysis: str
    metrics_summary: Dict[str, Any]


class SuggestArchitectureRequest(BaseModel):
    """Request body for architecture suggestion."""
    num_features: int = Field(..., ge=1, description="Number of input features")
    num_classes: int = Field(..., ge=2, description="Number of output classes")
    num_samples: int = Field(..., ge=1, description="Number of training samples")
    task_description: Optional[str] = Field(None, description="Optional task description")


class SuggestArchitectureResponse(BaseModel):
    """Response for architecture suggestion."""
    suggestion: str
    recommended_config: Optional[Dict[str, Any]] = None


class ExplainOptimizationRequest(BaseModel):
    """Request body for optimization explanation."""
    algorithm: str = Field(..., description="Algorithm used (pso or ga)")
    best_params: Dict[str, Any] = Field(default={}, description="Best parameters found")
    best_fitness: float = Field(..., ge=0, le=1, description="Best fitness (accuracy) achieved")
    iterations: int = Field(default=0, ge=0, description="Number of iterations completed")


class ExplainOptimizationResponse(BaseModel):
    """Response for optimization explanation."""
    explanation: str
    summary: Dict[str, Any]


class LLMStatusResponse(BaseModel):
    """Response for LLM status check."""
    status: str
    model: str
    base_url: str
    error: Optional[str] = None


# Endpoints

@router.post("/analyze-training", response_model=AnalyzeTrainingResponse)
async def analyze_training(request: AnalyzeTrainingRequest):
    """
    Analyze training results using LLM.
    
    Provides insights on:
    - Loss curve behavior
    - Overfitting detection
    - Performance assessment
    - Improvement suggestions
    
    Returns concise analysis (2-3 sentences) suitable for UI display.
    """
    try:
        llm_service = get_llm_service()
        
        # Convert to dict for the service
        metrics_dict = {
            "train_loss": request.metrics.train_loss,
            "val_loss": request.metrics.val_loss,
            "train_accuracy": request.metrics.train_accuracy,
            "val_accuracy": request.metrics.val_accuracy,
        }
        
        config_dict = {
            "learning_rate": request.config.learning_rate,
            "batch_size": request.config.batch_size,
            "epochs": request.config.epochs,
        }
        
        # Get analysis from LLM
        analysis = llm_service.analyze_training(metrics_dict, config_dict)
        
        # Calculate summary metrics
        summary = {}
        if request.metrics.train_loss:
            summary["final_train_loss"] = request.metrics.train_loss[-1]
            summary["initial_train_loss"] = request.metrics.train_loss[0]
        if request.metrics.val_loss:
            summary["final_val_loss"] = request.metrics.val_loss[-1]
            summary["initial_val_loss"] = request.metrics.val_loss[0]
        if request.metrics.train_accuracy:
            summary["final_train_accuracy"] = request.metrics.train_accuracy[-1]
        if request.metrics.val_accuracy:
            summary["final_val_accuracy"] = request.metrics.val_accuracy[-1]
            summary["best_val_accuracy"] = max(request.metrics.val_accuracy)
        summary["epochs_completed"] = len(request.metrics.train_loss)
        
        # Detect overfitting
        if request.metrics.train_loss and request.metrics.val_loss:
            loss_gap = request.metrics.val_loss[-1] - request.metrics.train_loss[-1]
            summary["loss_gap"] = loss_gap
            summary["overfitting_indicator"] = loss_gap > 0.1
        
        return AnalyzeTrainingResponse(
            analysis=analysis,
            metrics_summary=summary
        )
        
    except Exception as e:
        logger.error(f"Training analysis failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze training results: {str(e)}"
        )


@router.post("/suggest-architecture", response_model=SuggestArchitectureResponse)
async def suggest_architecture(request: SuggestArchitectureRequest):
    """
    Get LLM-based architecture suggestion for a dataset.
    
    Analyzes dataset characteristics and recommends:
    - Number of hidden layers
    - Layer sizes
    - Dropout rate
    
    Returns brief recommendation (2-3 sentences).
    """
    try:
        llm_service = get_llm_service()
        
        # Get suggestion from LLM
        suggestion = llm_service.suggest_architecture(
            num_features=request.num_features,
            num_classes=request.num_classes,
            num_samples=request.num_samples
        )
        
        # Also get structured recommendation using heuristics
        heuristic = llm_service._generate_architecture_fallback(
            num_features=request.num_features,
            num_classes=request.num_classes,
            num_samples=request.num_samples
        )
        
        # Generate recommended config
        if request.num_samples < 1000:
            recommended = {
                "hidden_layers": [32, 16],
                "dropout": 0.3,
                "input_size": request.num_features,
                "output_size": request.num_classes
            }
        elif request.num_samples < 10000:
            recommended = {
                "hidden_layers": [64, 32],
                "dropout": 0.2,
                "input_size": request.num_features,
                "output_size": request.num_classes
            }
        else:
            recommended = {
                "hidden_layers": [128, 64, 32],
                "dropout": 0.1,
                "input_size": request.num_features,
                "output_size": request.num_classes
            }
        
        # Adjust for high feature count
        if request.num_features > 50:
            recommended["hidden_layers"][0] = min(
                recommended["hidden_layers"][0] * 2, 
                256
            )
        
        return SuggestArchitectureResponse(
            suggestion=suggestion,
            recommended_config=recommended
        )
        
    except Exception as e:
        logger.error(f"Architecture suggestion failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suggest architecture: {str(e)}"
        )


@router.post("/explain-optimization", response_model=ExplainOptimizationResponse)
async def explain_optimization(request: ExplainOptimizationRequest):
    """
    Explain optimization results in simple terms.
    
    Translates technical optimization results into understandable insights
    for users unfamiliar with PSO or GA.
    
    Returns clear explanation (2-3 sentences).
    """
    try:
        llm_service = get_llm_service()
        
        # Validate algorithm
        if request.algorithm.lower() not in ["pso", "ga"]:
            raise HTTPException(
                status_code=400,
                detail="Algorithm must be 'pso' or 'ga'"
            )
        
        # Get explanation from LLM
        explanation = llm_service.explain_optimization(
            algorithm=request.algorithm.lower(),
            best_params=request.best_params,
            best_fitness=request.best_fitness,
            iterations=request.iterations
        )
        
        # Build summary
        algo_full_name = (
            "Particle Swarm Optimization" 
            if request.algorithm.lower() == "pso" 
            else "Genetic Algorithm"
        )
        
        summary = {
            "algorithm": request.algorithm.upper(),
            "algorithm_full_name": algo_full_name,
            "best_accuracy_percent": round(request.best_fitness * 100, 2),
            "iterations_completed": request.iterations,
            "performance_rating": (
                "excellent" if request.best_fitness >= 0.9 else
                "good" if request.best_fitness >= 0.75 else
                "moderate" if request.best_fitness >= 0.5 else
                "needs improvement"
            )
        }
        
        # Extract network config if present
        if "network_config" in request.best_params:
            summary["optimized_network"] = request.best_params["network_config"]
        elif "hidden_layers" in request.best_params:
            summary["optimized_network"] = request.best_params
        
        return ExplainOptimizationResponse(
            explanation=explanation,
            summary=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Optimization explanation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to explain optimization: {str(e)}"
        )


@router.get("/status", response_model=LLMStatusResponse)
async def get_llm_status():
    """
    Check LLM service status.
    
    Tests connection to Ollama and returns status information.
    """
    try:
        llm_service = get_llm_service()
        status = llm_service.check_connection()
        
        return LLMStatusResponse(
            status=status.get("status", "unknown"),
            model=status.get("model", "unknown"),
            base_url=status.get("base_url", "unknown"),
            error=status.get("error")
        )
        
    except Exception as e:
        logger.error(f"LLM status check failed: {e}")
        return LLMStatusResponse(
            status="error",
            model="unknown",
            base_url="unknown",
            error=str(e)
        )


@router.post("/chat")
async def chat_with_llm(prompt: str):
    """
    Simple chat endpoint for general LLM queries.
    
    Use for ad-hoc questions about neural networks and ML.
    """
    try:
        llm_service = get_llm_service()
        response = await llm_service.generate_response(prompt)
        
        return {
            "prompt": prompt,
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate response: {str(e)}"
        )