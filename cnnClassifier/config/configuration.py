from cnnClassifier.constants import *
import os
from cnnClassifier.utils.common import read_yaml, create_directories,save_json
from cnnClassifier.entity.config_entity import (DataIngestionConfig,
                                                PrepareBaseModelConfig,
                                                TrainingConfig,
                                                EvaluationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        if "prepare_model" not in self.config:
            raise KeyError("Key 'prepare_model' is missing from config.yaml!")
        config = self.config.prepare_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
        root_dir=Path(config.root_dir),
        base_model_path=Path(config.base_model_path),
        updated_base_model_path=Path(config.updated_base_model_path),
        params_image_size=[256, 256, 3],  # Updated for custom CNN
        params_learning_rate=self.params.LEARNING_RATE,
        params_include_top=False,  # Not needed for custom CNN
        params_weights=None,  # No pre-trained weights for custom CNN
        params_classes=self.params.CLASSES
    )

        return prepare_base_model_config
