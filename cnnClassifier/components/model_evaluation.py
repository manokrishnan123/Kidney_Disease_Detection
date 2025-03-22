from pathlib import Path
from cnnClassifier.constants import *
import tensorflow as tf
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.logger import logger
from sklearn.metrics import classification_report, confusion_matrix
import json
import seaborn as sns
import matplotlib.pyplot as plt






# Evaluation Class
class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.model = None
        self.valid_generator = None

    def load_model(self):
        """Load the trained model"""
        logger.info(f"Loading trained model from {self.config.trained_model_path}")
        self.model = tf.keras.models.load_model(self.config.trained_model_path)
        logger.info("Model loaded successfully")

    def create_validation_generator(self):
        """Create validation data generator"""
        logger.info("Creating validation data generator...")

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

        # Use the combined training data directory created during training
        combined_training_dir = "artifacts/data_ingestion/Combined_Training_Data"

        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=combined_training_dir,
            subset="validation",
            shuffle=False,
            **dataflow_kwargs
        )
        logger.info(f"Validation generator created with {self.valid_generator.samples} samples")

    def evaluate(self):
        """Evaluate the model and log metrics"""
        logger.info("Starting model evaluation...")

        # Ensure model and generator are initialized
        if self.model is None:
            self.load_model()
        if self.valid_generator is None:
            self.create_validation_generator()
            
        # Evaluate the model
        scores = self.model.evaluate(self.valid_generator)
        logger.info(f"Validation Loss: {scores[0]:.4f}")
        logger.info(f"Validation Accuracy: {scores[1]:.4f}")
        
        # Generate predictions
        predictions = self.model.predict(self.valid_generator)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = self.valid_generator.classes
        
        # Classification Report
        report = classification_report(y_true, y_pred, target_names=['Non-Stone', 'Stone'])
        logger.info("Classification Report:\n" + report)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        logger.info("Confusion Matrix:\n" + str(cm))
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # Save evaluation metrics
        metrics = {
            "loss": float(scores[0]),
            "accuracy": float(scores[1]),
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
        save_path = self.config.root_dir / "evaluation_metrics.json"
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved to {save_path}")

    def save_model(self, path: Path):
        """Save the evaluated model if needed"""
        self.model.save(path)
        logger.info(f"Model saved at {path}")
