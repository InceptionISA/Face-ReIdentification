from .providers import  MTCNNDetector, DlibDetector
from .FaceDetectionEnums import FaceDetectionEnums
from controllers.BaseController import BaseController

class FaceDetectorFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()
    
    def create(self, provider: str):

            
        if provider == FaceDetectionEnums.MTCNN.value:
            return MTCNNDetector(
                min_face_size=self.config.MTCNN_MIN_FACE_SIZE
            )
            
        elif provider == FaceDetectionEnums.DLIB.value:
            model_path = self.base_controller.get_model_path(
                self.config.DLIB_DETECTOR_MODEL_PATH
            )
            return DlibDetector(
                model_path=model_path
            )
        
        return None
    

    