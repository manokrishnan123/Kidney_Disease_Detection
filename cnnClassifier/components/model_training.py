import os
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from cnnClassifier.logger import logger  # ➕ Import your custom logger

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        """Load the base model for training"""
        logger.info("Loading base model...")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)  # ✅ Now loads base_model.h5
        logger.info("Base model loaded successfully.")


    def train_valid_generator(self):
        """Prepare data generators for training and validation"""
        logger.info("Initializing training and validation data generators...")  # ➕ Log message

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=40,
                horizontal_flip=True,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                **datagenerator_kwargs
            )
        else:
            train_datagenerator = valid_datagenerator

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        logger.info("Data generators initialized successfully.")  # ➕ Log message

    def train(self):
        """Train the model with logging and TensorBoard"""
        logger.info("Starting model training...")  # ➕ Log message

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Ensure log directory exists
        log_dir = Path(self.config.root_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # ➕ TensorBoard callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))

        try:
            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=[tensorboard_callback]  # ➕ TensorBoard logging
            )
            logger.info("Model training completed successfully.")  # ➕ Log message

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            logger.info(f"Model saved at {self.config.trained_model_path}")  # ➕ Log message

        except Exception as e:
            logger.error(f"Training failed due to: {e}")  # ➕ Log error
            raise e

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        """Save the trained model"""
        model.save(path)
