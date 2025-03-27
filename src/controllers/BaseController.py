from helpers.config import get_settings , Settings
import os
import random
import  string

class BaseController:
   
   def __init__(self):
        self.app_settings = get_settings()
        self.base_dir = os.path.dirname( os.path.dirname(__file__))
        self.file_dir = os.path.join(self.base_dir, 'assets/files')
        self.database_dir = os.path.join(self.base_dir, 'assets/database')


   def generate_random_string(self, length = 10):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))
   
   def get_database_path(self ,db_name):
        database_path = os.path.join(self.database_dir , db_name)

        if not os.path.exists(database_path):
            os.makedirs(database_path)

        return database_path
   
   
   def get_model_path(self, model_path):
        return os.path.join(self.base_dir, model_path)
   

   def get_image_path(self, project_id, person_id,):

     person_dir = os.path.join(self.file_dir, project_id, person_id)

     if not os.path.exists(person_dir):
        os.makedirs(person_dir)
     
     return person_dir
   
