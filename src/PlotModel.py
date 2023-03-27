import cv2
import keras as keras
import numpy as np


class PlotModel:
    def __init__(self):
        self.model = keras.models.load_model('models/mnist.h5')

    def predict(self, image):
        processed = PlotModel.keras_process_image(image)
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
