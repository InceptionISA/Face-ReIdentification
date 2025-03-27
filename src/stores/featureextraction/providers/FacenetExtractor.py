import numpy as np
from ..FeatureExtractorInterface import FeatureExtractorInterface

from typing import List, Optional
import os

class FacenetExtractor(FeatureExtractorInterface):
    def __init__(self, model_path: Optional[str] = None, input_size: tuple = (160, 160), normalization: str = "per_image"):
        self.model_path = model_path
        self.input_size = input_size
        self.normalization = normalization
        self.model = None

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = "Facenet model loaded"


    def unload_model(self):
        self.model = None

    