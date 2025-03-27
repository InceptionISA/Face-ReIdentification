from ..FaceDetectorInterface import FaceDetectorInterface
from typing import Optional

class DlibDetector(FaceDetectorInterface):
    def __init__(self, model_path: str):
        self.model_path = model_path


    