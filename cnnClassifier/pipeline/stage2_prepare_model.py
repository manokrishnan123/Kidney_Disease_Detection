from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_model import PrepareBaseModel
from cnnClassifier.logger import logger

STAGE_NAME = "Prepare base model"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        

        # Create and save the custom CNN model
        prepare_base_model.get_base_model()
        logger.info("Custom CNN model created successfully.")

        # Save the updated model
        prepare_base_model.update_base_model()
        logger.info("Custom CNN model saved successfully.")

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
