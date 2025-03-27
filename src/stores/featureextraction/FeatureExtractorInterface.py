from typing import List, Dict, Optional
import numpy as np
from abc import ABC, abstractmethod


class FeatureExtractorInterface(ABC):
    """Interface for feature extraction models"""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the feature extractor.

        Args:
            model_path (Optional[str]): Path to the feature extraction model.
            kwargs: Additional parameters for customization.
        """
        self.model_path = model_path
        self.config = kwargs  # Store additional parameters
        self.model = None  # Placeholder for the loaded model

    # @abstractmethod
    # def initialize(self) -> None:
    #     """Initialize the feature extraction model (e.g., load weights)."""
    #     pass

    # @abstractmethod
    # def load_model(self, model_path: str) -> None:
    #     """Load the feature extraction model from the specified path."""
    #     pass

    # @abstractmethod
    # def unload_model(self) -> None:
    #     """Unload the feature extraction model to free resources."""
    #     pass

    # @abstractmethod
    # def extract_features(self, image_path: str, bbox: Dict) -> np.ndarray:
    #     """
    #     Extract face embeddings from an image.

    #     Args:
    #         image_path (str): Path to the input image.
    #         bbox (Dict): Bounding box of the detected face.

    #     Returns:
    #         np.ndarray: Face embedding vector.
    #     """
    #     pass

    # @abstractmethod
    # def set_feature_dimension(self, dimension: int) -> None:
    #     """Set the output feature vector dimension."""
    #     pass
