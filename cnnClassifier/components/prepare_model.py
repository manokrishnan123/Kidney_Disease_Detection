import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from pathlib import Path
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    def get_base_model(self):
        """Defines and saves the custom CNN model."""
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=(256,256,3)),
            MaxPooling2D(),
            Conv2D(32, (3,3), activation='relu'),
            MaxPooling2D(),
            Conv2D(16, (3,3), activation='relu'),
            MaxPooling2D(),
            Flatten(),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        self.save_model(path=self.config.base_model_path, model=model)

   ## def update_base_model(self):
        #"""Loads and saves the model (if needed)."""
        #model = tf.keras.models.load_model(self.config.base_model_path)
        #self.save_model(path=self.config.updated_base_model_path, model=model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
