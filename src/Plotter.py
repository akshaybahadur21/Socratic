from HandDetector import HandDetector
from utils.VisionUtils import get_idx_to_coordinates, rescale_frame
import matplotlib
from src.PlotModel import PlotModel

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import cv2
import numpy as np

fig = plt.figure()
plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.axvline(x=0, c="black", label="x=0")
plt.axhline(y=0, c="black", label="y=0")
plt.plot(0, 0, marker="o", markersize=5, markeredgecolor="green", markerfacecolor="green")
x_plot = np.linspace(-5.0, 5.0, num=10)

class Plotter:
    def _init_(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)
        self.detector = HandDetector()
        self.plot_model = PlotModel()
        self.brush_thick = 15
        self.eraser_thick = 40
        self.rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
        self.options = "--psm 8"

    def plot_equation(self):
        xp, yp = 0, 0
        blkboard = np.zeros((720, 1280, 3), np.uint8)
        count = 0
        while True:
            _, img = self.cap.read()
            if img is None:
                continue
            img = cv2.flip(img, 1)
            img = cv2.putText(img, "y=", (800, 250), cv2.FONT_HERSHEY_TRIPLEX, color = (255, 0, 255), fontScale=2, thickness=3)
            img = cv2.rectangle(img, (910, 290), (1000, 190), (0, 255, 0), 2)
            img = cv2.putText(img, "x +", (1020, 250), cv2.FONT_HERSHEY_TRIPLEX, color = (255, 0, 255), fontScale=2, thickness=3)
            img = cv2.rectangle(img, (1150, 290), (1240, 190), (0, 255, 0), 2)

            img = self.detector.find_hands(img)
            landmark_list = self.detector.find_position(img, draw=False)

            if len(landmark_list) != 0:

                _, x1, y1 = landmark_list[8]  # Tip of Index Finger
                _, x2, y2 = landmark_list[12]  # Tip of Middle Finger
                fingers = self.detector.fingers()

                if len(fingers) == 5:
                    if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                        cv2.rectangle(img, (x1 - 25, y1 - 25), (x2 + 25, y2 + 25), (0, 0, 255), cv2.FILLED)
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        # cv2.line(img, (xp, yp), (x1, y1), (0, 0, 0), self.eraser_thick)
                        cv2.line(blkboard, (xp, yp), (x1, y1), (0, 0, 0), self.eraser_thick)
                        xp, yp = x1, y1

                    elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[
                        4] == 0:
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        print("write")
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                        # cv2.line(img, (xp, yp), (x1, y1), (0, 255, 0), self.brush_thick)
                        cv2.line(blkboard, (xp, yp), (x1, y1), (0, 255, 0), self.brush_thick)
                        xp, yp = x1, y1

                    elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[
                        4] == 0:
                        xp, yp = 0, 0
                        print("Go")

                        blackboard_gray = cv2.cvtColor(blkboard, cv2.COLOR_RGB2GRAY)
                        blur1 = cv2.medianBlur(blackboard_gray, 15)
                        blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                        thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                        blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                                              cv2.CHAIN_APPROX_SIMPLE)
                        if blackboard_cnts is None:
                            continue
                        if len(blackboard_cnts) == 2:
                            for cnt in blackboard_cnts:
                                if cv2.contourArea(cnt) > 500:
                                    x, y, w, h = cv2.boundingRect(cnt)
                                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255))
                                    if abs(x - 910) <= 50:
                                        digit = blackboard_gray[y:y + h, x:x + w]
                                        pred_probab, m = self.math_model.predict(digit)
                                    if abs(x - 1150) <= 50:
                                        digit = blackboard_gray[y:y + h, x:x + w]
                                        pred_probab, c = self.math_model.predict(digit)
                            plt.plot(x_plot, m * x_plot + c
                                     , linestyle='solid', color="blue")
                            plt.title("Socratic Plot for y = {}x + {}".format(m, c))
                            plt.xlabel("X")
                            plt.ylabel("Y")

                            fig.canvas.draw()
                            cnv = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                                                sep='')
                            cnv = cnv.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                            cnv = cv2.cvtColor(cnv, cv2.COLOR_RGB2BGR)
                            cv2.imshow("plot", cnv)
                    else:
                        xp, yp = 0, 0
                        cv2.destroyWindow("plot")

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(blkboard, cv2.COLOR_BGR2GRAY)
            _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, inv)
            img = cv2.bitwise_or(img, blkboard)

            cv2.imshow("Frame", rescale_frame(img, 200))
            k = cv2.waitKey(33) & 0xFF
            if k == 27:
                break