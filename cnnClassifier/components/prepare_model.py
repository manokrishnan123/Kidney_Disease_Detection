import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from cnnClassifier.logger import logger  

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Defines and saves the custom CNN model."""
        model = Sequential([
        # Block 1
        Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)),
        BatchNormalization(), #stabilization
        MaxPooling2D(), #downsample feature map

        # Block 2
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),

        # Block 3
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.3), #prevent overfitting

        # Block 4
        Conv2D(256, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(),
        Dropout(0.4),

        # Dense Classification Head
        Flatten(),
        Dense(256, activation='relu', kernel_initializer="he_normal"),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification
    ])

        # Save the model using the newly added method
        self.save_model(path=self.config.base_model_path, model=model)

    def save_model(self, path: Path, model: tf.keras.Model):
        """Save the trained model to a given path."""
        model.save(path)
        logger.info(f"Model saved successfully at: {path}")

            
            
