import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        img = image.load_img(self.filename, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # normalize like in training

        prediction_score = model.predict(img_array)[0][0]
        print("Prediction score:", prediction_score)

        if prediction_score > 0.5:
            result = "Kidney with Stone"
        else:
            result = "Normal Kidney without stone"

        return [{"image": result}]
