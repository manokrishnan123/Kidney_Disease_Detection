import os
import shutil
import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import TrainingConfig
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from cnnClassifier.logger import logger 
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import matplotlib.pyplot as plt


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    def get_base_model(self):
        
        """Load and compile the base model for training"""
        logger.info("Loading base model...")
        self.model = tf.keras.models.load_model(self.config.updated_base_model_path)

        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        logger.info("Base model loaded and compiled successfully.")



    def train_valid_generator(self):
        """Prepare data generators for training and validation"""
        logger.info("Initializing training and validation data generators...")

        datagenerator_kwargs = dict(
            rescale=1./255,
            validation_split=0.20
        )

        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],
            batch_size=self.config.params_batch_size,
            class_mode="binary",  
            interpolation="bilinear"
        )

        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
            **datagenerator_kwargs
        )

        # Path to both datasets
        dataset_paths = [
            os.path.join(self.config.training_data, "Original Dataset"),
            os.path.join(self.config.training_data, "Augmented Dataset")
        ]

        # Create a combined dataset folder
        combined_training_dir = "artifacts/data_ingestion/Combined_Training_Data"

        if not os.path.exists(combined_training_dir):
            os.makedirs(combined_training_dir, exist_ok=True)
            os.makedirs(os.path.join(combined_training_dir, "Stone"), exist_ok=True)
            os.makedirs(os.path.join(combined_training_dir, "Non-Stone"), exist_ok=True)

        # Copy images from BOTH datasets
        for dataset_path in dataset_paths:
            for class_name in ["Stone", "Non-Stone"]:
                src_folder = os.path.join(dataset_path, class_name)
                dest_folder = os.path.join(combined_training_dir, class_name)

                if os.path.exists(src_folder):
                    for file in os.listdir(src_folder):
                        src_file_path = os.path.join(src_folder, file)
                        dest_file_path = os.path.join(dest_folder, file)

                        # Avoid overwriting duplicate files by renaming
                        if os.path.exists(dest_file_path):
                            filename, ext = os.path.splitext(file)
                            dest_file_path = os.path.join(dest_folder, f"{filename}_copy{ext}")

                        shutil.copy(src_file_path, dest_file_path)

        # Load from the newly created combined dataset
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=combined_training_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )

        if self.config.params_is_augmentation:
            train_datagenerator = ImageDataGenerator(
                rescale=1./255,
                rotation_range=20,
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                horizontal_flip=True,
                validation_split=0.20
            )
        else:
            train_datagenerator = ImageDataGenerator(
                rescale=1./255,
                validation_split=0.20
            )

        self.train_generator = train_datagenerator.flow_from_directory(
            directory=combined_training_dir,
            subset="training",
            shuffle=True,
            **dataflow_kwargs
        )

        # Print dataset info
        print("Training data classes:", self.train_generator.class_indices)
        print("Number of training samples:", self.train_generator.samples)
        print("Number of validation samples:", self.valid_generator.samples)

        logger.info("Data generators initialized successfully.")





    def train(self):
        """Train the model with logging, Early Stopping, and Class Weights"""
        logger.info("Starting model training...")

        self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
        self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

        # Ensure log directory exists
        log_dir = Path(self.config.root_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Look in BOTH "Original Dataset" and "Augmented Dataset"
        stone_count = 0
        non_stone_count = 0
        dataset_paths = [
            os.path.join(self.config.training_data, "Original Dataset"),
            os.path.join(self.config.training_data, "Augmented Dataset")
        ]

        for dataset_path in dataset_paths:
            if os.path.exists(dataset_path):
                stone_count += len(os.listdir(os.path.join(dataset_path, "Stone")))
                non_stone_count += len(os.listdir(os.path.join(dataset_path, "Non-Stone")))

        # Handle cases where folders don't exist
        if stone_count == 0 or non_stone_count == 0:
            raise FileNotFoundError(f"One of the class folders (Stone or Non-Stone) is missing in {dataset_paths}")

        # Compute class weights
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=np.array([0, 1]),
            y=[0] * non_stone_count + [1] * stone_count
        )

        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        # Early Stopping Callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        )

        # TensorBoard Callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(log_dir))

        # Model Checkpoint Callback to save the best model during training
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.config.trained_model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        )

        try:
            history = self.model.fit(
            self.train_generator,
            epochs=self.config.params_epochs,
            steps_per_epoch=self.steps_per_epoch,
            validation_steps=self.validation_steps,
            validation_data=self.valid_generator,
            class_weight=class_weight_dict,
            callbacks=[early_stopping, tensorboard_callback, checkpoint]
        )
            
            
            logger.info("Model training completed successfully.")

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
            logger.info(f"Model saved at {self.config.trained_model_path}")

        except Exception as e:
            logger.error(f"Training failed due to: {e}")
            raise e
        
        # Plot training & validation accuracy/loss
        try:
            metrics_dir = Path(self.config.root_dir) / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            # Plot Accuracy
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.title('Model Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(metrics_dir / 'accuracy.png')
            plt.close()

            # Plot Loss
            plt.figure(figsize=(8, 6))
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(metrics_dir / 'loss.png')
            plt.close()

            logger.info(f"Training metrics plotted and saved to {metrics_dir}")

        except Exception as plot_err:
            logger.warning(f"Could not plot training metrics due to: {plot_err}")

        
    def save_model(self, path: Path, model: tf.keras.Model):
        """Save the trained model to a given path."""
        model.save(path)
        logger.info(f"Model saved successfully at: {path}")   