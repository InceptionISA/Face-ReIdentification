from ..FaceDetectorInterface import FaceDetectorInterface
from typing import Optional

class MTCNNDetector (FaceDetectorInterface):
    def __init__(self, min_face_size: int):
        self.min_face_size = min_face_size


        