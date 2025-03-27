from controllers.BaseController import BaseController
from fastapi import UploadFile
from .ProjectController import ProjectController
from models import ResponseMessage
import re
import os


class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1024 * 1024  # Convert MB to bytes

    def validate_uploaded_file(self, file: UploadFile):

        if file.content_type not in self.app_settings.FILE_ALLOWED_TYPES:
            return False, ResponseMessage.FILETYPENOTSUPPORTED.value

        if file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale:
            return False, ResponseMessage.FILESIZEEXCEEDED.value

        return True, ResponseMessage.FILEUPLOADSUCCESS.value

    def generate_unique_file_path(self, file_name: str, project_id: str, person_id: str):

        project_dir_path = ProjectController().get_project_path(project_id)
        person_dir_path = os.path.join(project_dir_path, person_id)

        # Ensure person directory exists
        os.makedirs(person_dir_path, exist_ok=True)

        cleaned_file_name = self.get_cleaned_file_name(file_name)

        while True:
            random_file_name = self.generate_random_string()
            new_file_path = os.path.join(person_dir_path, f"{random_file_name}_{cleaned_file_name}")
            if not os.path.exists(new_file_path):
                return new_file_path, f"{random_file_name}_{cleaned_file_name}"

    def get_cleaned_file_name(self, file_name: str):
        """Clean file name to remove special characters except dots and spaces."""
        return re.sub(r'[^\w\s.-]', '', file_name.strip())  # Allows dots and hyphens

