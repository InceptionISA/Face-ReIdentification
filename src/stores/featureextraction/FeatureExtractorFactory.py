from .providers import FacenetExtractor
from .FeatureExtractionEnums import FeatureExtractionEnums
from controllers.BaseController import BaseController

class FeatureExtractorFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        if provider == FeatureExtractionEnums.FACENET.value:
            model_path = self.base_controller.get_model_path(
                self.config.FACENET_MODEL_PATH
            )
            return FacenetExtractor(
                model_path=model_path
            )

        
        return None