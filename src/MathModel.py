import math

import cv2
import keras as keras
import numpy as np


class MathModel:
    def __init__(self):
        self.model = keras.models.load_model('models/socratic.h5')
        self.class_dict = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                           9: '9', 10: '+', 11: '-', 12: '*', 13: '/', 14: math.pi}

    def predict(self, image):
        processed = MathModel.keras_process_image(image)
        pred_probab = self.model.predict(processed)[0]
        pred_class = list(pred_probab).index(max(pred_probab))
        return max(pred_probab), self.class_dict[pred_class]

    @staticmethod
    def keras_process_image(img):
        image_x = 28
        image_y = 28
        img = cv2.resize(img, (image_x, image_y))
        img = np.array(img, dtype=np.float32)
        # img = img.astype("float32") / 255
        img = np.reshape(img, (-1, image_x, image_y, 1))
        return img
