"""
LLM Service for Neural Network Analysis and Suggestions.

This module provides LLM-powered insights using Ollama with llama3.2:1b model.
Features:
- Training results analysis (overfitting detection, performance insights)
- Architecture suggestions based on dataset characteristics
- Optimization results explanation

Uses LangChain for prompt management and Ollama for local LLM inference.

Author: NN Optimizer Project
"""

import json
import re
import logging
from typing import Dict, Any, Optional

from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

from app.config import settings
from app.schemas import NetworkConfig, NetworkSuggestionResponse

logger = logging.getLogger(__name__)


class LLMService:
    """
    LLM service using Ollama for neural network insights.
    Uses llama3.2:1b (small, fast model) for quick responses.
    
    Features:
    - Training analysis with overfitting detection
    - Dataset-based architecture suggestions
    - Optimization results explanation
    - Network architecture recommendations
    
    All responses are designed to be concise (2-3 sentences) for UI display.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the LLM service with Ollama.
        
        Args:
            base_url: Ollama server URL (defaults to settings.ollama_base_url)
            model: Model name (defaults to settings.ollama_model)
        """
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        
        self.llm = Ollama(
            base_url=self.base_url,
            model=self.model,
            temperature=0.3,  # Lower temperature for more consistent responses
        )
        
        logger.info(f"LLM Service initialized with model {self.model} at {self.base_url}")

    def _safe_invoke(self, prompt: str, fallback: str = "Analysis unavailable.") -> str:
        """
        Safely invoke the LLM with error handling.
        
        Args:
            prompt: The prompt to send to the LLM
            fallback: Message to return if LLM fails
            
        Returns:
            LLM response or fallback message
        """
        try:
            response = self.llm.invoke(prompt)
            return response.strip()
        except Exception as e:
            logger.warning(f"LLM invocation failed: {e}")
            return fallback

    def analyze_training(self, metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
        """
        Analyze training results and provide insights.
        
        Analyzes loss curves, accuracy progression, and identifies potential issues
        like overfitting. Provides actionable improvement suggestions.
        
        Args:
            metrics: Training metrics dict with keys:
                - train_loss: List of training losses per epoch
                - val_loss: List of validation losses per epoch
                - train_accuracy: List of training accuracies (0-1 scale)
                - val_accuracy: List of validation accuracies (0-1 scale)
            config: Training configuration dict with keys:
                - learning_rate: Learning rate used
                - batch_size: Batch size used
                - epochs: Total epochs configured
                
        Returns:
            Concise analysis string (2-3 sentences)
        """
        # Extract final metrics safely
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])
        train_accuracy = metrics.get("train_accuracy", [])
        val_accuracy = metrics.get("val_accuracy", [])
        
        if not train_loss or not val_loss:
            return "Insufficient training data for analysis. Please complete a training run first."
        
        final_train_loss = train_loss[-1] if train_loss else 0
        final_val_loss = val_loss[-1] if val_loss else 0
        final_train_acc = (train_accuracy[-1] * 100) if train_accuracy else 0
        final_val_acc = (val_accuracy[-1] * 100) if val_accuracy else 0
        epochs_completed = len(train_loss)
        
        # Calculate overfitting indicator
        loss_gap = final_val_loss - final_train_loss
        acc_gap = final_train_acc - final_val_acc
        
        prompt_template = PromptTemplate(
            input_variables=[
                "final_train_loss", "final_val_loss",
                "final_train_acc", "final_val_acc", 
                "epochs", "loss_gap", "acc_gap",
                "learning_rate"
            ],
            template="""Analyze these neural network training results briefly:
- Training Loss: {final_train_loss:.4f}, Validation Loss: {final_val_loss:.4f}
- Training Accuracy: {final_train_acc:.1f}%, Validation Accuracy: {final_val_acc:.1f}%
- Epochs trained: {epochs}
- Loss gap (val-train): {loss_gap:.4f}
- Accuracy gap (train-val): {acc_gap:.1f}%
- Learning rate: {learning_rate}

In 2-3 sentences, comment on model performance and any signs of overfitting. Suggest one improvement if needed."""
        )
        
        prompt = prompt_template.format(
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            final_train_acc=final_train_acc,
            final_val_acc=final_val_acc,
            epochs=epochs_completed,
            loss_gap=loss_gap,
            acc_gap=acc_gap,
            learning_rate=config.get("learning_rate", 0.001)
        )
        
        # Fallback analysis based on simple heuristics
        fallback = self._generate_training_fallback(
            final_train_acc, final_val_acc, loss_gap, acc_gap
        )
        
        return self._safe_invoke(prompt, fallback)

    def _generate_training_fallback(
        self, 
        train_acc: float, 
        val_acc: float, 
        loss_gap: float,
        acc_gap: float
    ) -> str:
        """Generate heuristic-based training analysis as fallback."""
        insights = []
        
        # Check for overfitting
        if loss_gap > 0.2 or acc_gap > 10:
            insights.append("Signs of overfitting detected (training metrics significantly better than validation).")
            insights.append("Consider adding more dropout, reducing model complexity, or using data augmentation.")
        elif val_acc >= 90:
            insights.append(f"Excellent performance with {val_acc:.1f}% validation accuracy.")
            insights.append("The model generalizes well to unseen data.")
        elif val_acc >= 70:
            insights.append(f"Good performance with {val_acc:.1f}% validation accuracy.")
            insights.append("Consider training longer or tuning hyperparameters for improvement.")
        else:
            insights.append(f"Model achieves {val_acc:.1f}% validation accuracy.")
            insights.append("Consider a larger model, different learning rate, or more training data.")
        
        return " ".join(insights)

    def suggest_architecture(
        self, 
        num_features: int, 
        num_classes: int, 
        num_samples: int
    ) -> str:
        """
        Suggest network architecture based on dataset characteristics.
        
        Provides recommendations for hidden layer configuration based on
        dataset size and complexity.
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes
            num_samples: Number of training samples
            
        Returns:
            Architecture suggestion string (2-3 sentences)
        """
        prompt_template = PromptTemplate(
            input_variables=["num_features", "num_classes", "num_samples"],
            template="""For a classification dataset with:
- {num_features} input features
- {num_classes} classes
- {num_samples} samples

Suggest a simple neural network architecture. Recommend number of hidden layers 
and approximate sizes. Keep response to 2-3 sentences with specific numbers."""
        )
        
        prompt = prompt_template.format(
            num_features=num_features,
            num_classes=num_classes,
            num_samples=num_samples
        )
        
        # Generate fallback suggestion
        fallback = self._generate_architecture_fallback(
            num_features, num_classes, num_samples
        )
        
        return self._safe_invoke(prompt, fallback)

    def _generate_architecture_fallback(
        self, 
        num_features: int, 
        num_classes: int, 
        num_samples: int
    ) -> str:
        """Generate heuristic-based architecture suggestion as fallback."""
        # Simple heuristics based on dataset size
        if num_samples < 1000:
            layers = [32, 16]
            dropout = 0.3
            reasoning = "Small dataset: use a shallow network with 2 hidden layers (32, 16 neurons) and 30% dropout to prevent overfitting."
        elif num_samples < 10000:
            layers = [64, 32]
            dropout = 0.2
            reasoning = "Medium dataset: recommend 2 hidden layers (64, 32 neurons) with 20% dropout for balanced capacity."
        else:
            layers = [128, 64, 32]
            dropout = 0.1
            reasoning = "Large dataset: can support 3 hidden layers (128, 64, 32 neurons) with 10% dropout."
        
        # Adjust for feature count
        if num_features > 50:
            layers[0] = min(layers[0] * 2, 256)
            reasoning += f" First layer expanded due to {num_features} input features."
        
        return reasoning

    def explain_optimization(
        self, 
        algorithm: str, 
        best_params: Dict[str, Any], 
        best_fitness: float,
        iterations: int = 0
    ) -> str:
        """
        Explain optimization results in simple terms.
        
        Translates technical optimization results into understandable insights.
        
        Args:
            algorithm: Optimization algorithm used ('pso' or 'ga')
            best_params: Best hyperparameters found
            best_fitness: Best fitness (accuracy) achieved
            iterations: Number of iterations completed
            
        Returns:
            Explanation string (2-3 sentences)
        """
        algo_name = "Particle Swarm Optimization" if algorithm == "pso" else "Genetic Algorithm"
        
        prompt_template = PromptTemplate(
            input_variables=[
                "algorithm", "algo_name", "best_fitness", 
                "iterations", "best_params"
            ],
            template="""Explain these {algo_name} ({algorithm}) optimization results simply:
- Best accuracy achieved: {best_fitness:.1f}%
- Iterations completed: {iterations}
- Best parameters found: {best_params}

In 2-3 sentences, explain what the optimizer found and whether the results look good."""
        )
        
        prompt = prompt_template.format(
            algorithm=algorithm.upper(),
            algo_name=algo_name,
            best_fitness=best_fitness * 100,
            iterations=iterations,
            best_params=json.dumps(best_params, indent=2)
        )
        
        # Generate fallback explanation
        fallback = self._generate_optimization_fallback(
            algorithm, best_params, best_fitness
        )
        
        return self._safe_invoke(prompt, fallback)

    def _generate_optimization_fallback(
        self, 
        algorithm: str, 
        best_params: Dict[str, Any],
        best_fitness: float
    ) -> str:
        """Generate heuristic-based optimization explanation as fallback."""
        algo_name = "Particle Swarm Optimization" if algorithm == "pso" else "Genetic Algorithm"
        accuracy = best_fitness * 100
        
        if accuracy >= 90:
            quality = "excellent"
        elif accuracy >= 75:
            quality = "good"
        elif accuracy >= 50:
            quality = "moderate"
        else:
            quality = "limited"
        
        explanation = f"{algo_name} found a configuration achieving {accuracy:.1f}% accuracy, which is {quality} performance. "
        
        if "hidden_layers" in best_params or "network_config" in best_params:
            config = best_params.get("network_config", best_params)
            layers = config.get("hidden_layers", [])
            if layers:
                explanation += f"The optimized network uses {len(layers)} hidden layer(s) with sizes {layers}."
        
        return explanation

    async def suggest_network_architecture(
        self,
        num_features: int,
        num_classes: int,
        num_samples: int,
        task_description: Optional[str] = None,
    ) -> NetworkSuggestionResponse:
        """
        Use LLM to suggest a neural network architecture.
        
        This method is used by the /api/networks/suggest endpoint.
        
        Args:
            num_features: Number of input features
            num_classes: Number of output classes  
            num_samples: Number of training samples
            task_description: Optional description of the task
            
        Returns:
            NetworkSuggestionResponse with config and reasoning
        """
        prompt = f"""You are a neural network architecture expert. Based on the following dataset characteristics, suggest an optimal neural network architecture.

Dataset Information:
- Number of features (inputs): {num_features}
- Number of classes (outputs): {num_classes}
- Number of samples: {num_samples}
{f"- Task description: {task_description}" if task_description else ""}

Please suggest a neural network architecture. Respond with a JSON object containing:
1. "hidden_layers": an array of integers representing the number of neurons in each hidden layer
2. "dropout": a float between 0 and 0.5 for dropout rate
3. "reasoning": a brief explanation of your choices

Consider:
- Dataset size when determining network complexity
- Number of features when choosing input layer connections
- Number of classes for the output layer
- Appropriate dropout for regularization

Respond ONLY with the JSON object, no additional text.
"""

        try:
            response = self.llm.invoke(prompt)

            # Parse JSON from response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                suggestion = json.loads(json_match.group())
            else:
                # Default suggestion if parsing fails
                suggestion = {
                    "hidden_layers": [64, 32],
                    "dropout": 0.2,
                    "reasoning": "Default architecture due to parsing error.",
                }

            hidden_layers = suggestion.get("hidden_layers", [64, 32])
            dropout = suggestion.get("dropout", 0.2)
            reasoning = suggestion.get("reasoning", "LLM-suggested architecture.")

            # Validate and clamp values
            hidden_layers = [max(8, min(512, int(h))) for h in hidden_layers]
            dropout = max(0.0, min(0.5, float(dropout)))

            config = NetworkConfig(
                input_size=num_features,
                hidden_layers=hidden_layers,
                output_size=num_classes,
                dropout=dropout,
            )

            return NetworkSuggestionResponse(
                suggested_config=config,
                reasoning=reasoning,
            )

        except Exception as e:
            logger.warning(f"LLM suggestion failed: {e}")
            # Fallback to heuristic-based suggestion
            return self._heuristic_suggestion(num_features, num_classes, num_samples)

    def _heuristic_suggestion(
        self,
        num_features: int,
        num_classes: int,
        num_samples: int,
    ) -> NetworkSuggestionResponse:
        """Fallback heuristic-based architecture suggestion."""
        # Simple heuristics
        if num_samples < 1000:
            hidden_layers = [32, 16]
            dropout = 0.3
            reasoning = "Small dataset: using smaller network with higher dropout to prevent overfitting."
        elif num_samples < 10000:
            hidden_layers = [64, 32]
            dropout = 0.2
            reasoning = "Medium dataset: balanced architecture with moderate dropout."
        else:
            hidden_layers = [128, 64, 32]
            dropout = 0.1
            reasoning = "Large dataset: deeper network with lower dropout."

        # Adjust first layer based on features
        if num_features > 100:
            hidden_layers[0] = min(hidden_layers[0] * 2, 256)
            reasoning += f" Increased first layer size due to high feature count ({num_features})."

        config = NetworkConfig(
            input_size=num_features,
            hidden_layers=hidden_layers,
            output_size=num_classes,
            dropout=dropout,
        )

        return NetworkSuggestionResponse(
            suggested_config=config,
            reasoning=reasoning,
        )

    async def generate_response(self, prompt: str) -> str:
        """
        Generate a general LLM response.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response string
        """
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def check_connection(self) -> Dict[str, Any]:
        """
        Check if the LLM service is available.
        
        Returns:
            Dict with status and model info
        """
        try:
            # Simple test prompt
            response = self.llm.invoke("Say 'ok' if you're working.")
            return {
                "status": "connected",
                "model": self.model,
                "base_url": self.base_url,
                "test_response": response[:50] if response else None
            }
        except Exception as e:
            return {
                "status": "disconnected",
                "model": self.model,
                "base_url": self.base_url,
                "error": str(e)
            }


# Singleton instance for reuse
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get or create the singleton LLM service instance.
    
    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def reset_llm_service():
    """Reset the singleton instance (useful for testing)."""
    global _llm_service
    _llm_service = None