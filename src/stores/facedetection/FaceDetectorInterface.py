from typing import List, Dict, Optional
import numpy as np
from abc import ABC, abstractmethod

class FaceDetectorInterface(ABC):
    """Interface for face detection models"""

    def __init__(self, model_path: Optional[str] = None, **kwargs):
        """
        Initialize the face detector.

        Args:
            model_path (Optional[str]): Path to the detection model.
            kwargs: Additional parameters for customization.
        """
        self.model_path = model_path
        self.config = kwargs  # Store additional parameters
        self.model = None  # Placeholder for the loaded model

    # @abstractmethod
    # def initialize(self) -> None:
    #     """Initialize the face detection model (e.g., load weights)."""
    #     pass

    # @abstractmethod
    # def load_model(self, model_path: str) -> None:
    #     """Load the face detection model from the specified path."""
    #     pass

    # @abstractmethod
    # def unload_model(self) -> None:
    #     """Unload the face detection model to free resources."""
    #     pass

    # @abstractmethod
    # def detect_faces(self, image_path: str) -> List[Dict]:
    #     """
    #     Detect faces in an image.

    #     Args:
    #         image_path (str): Path to the input image.

    #     Returns:
    #         List[Dict]: List of detected faces with bounding boxes and other metadata.
    #     """
    #     pass

    # @abstractmethod
    # def set_detection_threshold(self, threshold: float) -> None:
    #     """Set the confidence threshold for face detection."""
    #     pass


