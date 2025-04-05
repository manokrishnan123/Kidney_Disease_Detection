import os
import zipfile
import shutil
import gdown
from cnnClassifier.logger import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import (DataIngestionConfig)



class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    
    def download_file(self)-> str:
        '''
        Fetch data from the url
        '''

        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            gdown.download(prefix+file_id,zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            raise e
        
    

    def extract_zip_file(self):
        """
        zip_file_path: str
        Extracts the zip file into the data directory
        Function returns None
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)


    def merge_original_and_augmented_datasets(self):
        """
        Merge 'Original Dataset' and 'Augmented Dataset' into a single 'Combined_Training_Data' folder.
        """

        dataset_root = os.path.join(self.config.unzip_dir, "Kidney Stone Dataset")
        dataset_paths = [
            os.path.join(dataset_root, "Original Dataset"),
            os.path.join(dataset_root, "Augmented Dataset")
        ]


        combined_training_dir = os.path.join("artifacts", "data_ingestion", "Combined_Training_Data")

        if not os.path.exists(combined_training_dir):
            os.makedirs(combined_training_dir, exist_ok=True)
            os.makedirs(os.path.join(combined_training_dir, "Stone"), exist_ok=True)
            os.makedirs(os.path.join(combined_training_dir, "Non-Stone"), exist_ok=True)

        for dataset_path in dataset_paths:
            for class_name in ["Stone", "Non-Stone"]:
                src_folder = os.path.join(dataset_path, class_name)
                dest_folder = os.path.join(combined_training_dir, class_name)

                if os.path.exists(src_folder):
                    for file in os.listdir(src_folder):
                        src_file_path = os.path.join(src_folder, file)
                        dest_file_path = os.path.join(dest_folder, file)

                        if os.path.exists(dest_file_path):
                            filename, ext = os.path.splitext(file)
                            dest_file_path = os.path.join(dest_folder, f"{filename}_copy{ext}")

                        shutil.copy(src_file_path, dest_file_path)
