import json
import re
from typing import Dict, Any, Optional
from langchain_community.llms import Ollama

from app.config import settings
from app.schemas import NetworkConfig, NetworkSuggestionResponse


class LLMService:
    """Service for interacting with local LLM via Ollama."""

    def __init__(self):
        self.llm = Ollama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
        )

    async def suggest_network_architecture(
        self,
        num_features: int,
        num_classes: int,
        num_samples: int,
        task_description: Optional[str] = None,
    ) -> NetworkSuggestionResponse:
        """Use LLM to suggest a neural network architecture."""
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
        """Generate a general LLM response."""
        try:
            return self.llm.invoke(prompt)
        except Exception as e:
            return f"Error generating response: {str(e)}"
