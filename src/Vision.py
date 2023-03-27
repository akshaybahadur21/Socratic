import numpy as np
import cv2
from collections import deque
import mediapipe as mp

from src.MathModel import MathModel
from src.utils.VisionUtils import get_idx_to_coordinates, rescale_frame
from src.utils.MathUtils import solve_eqn


class Vision:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.math_model = MathModel()
        self.x = 600
        self.y = 50
        self.w = 650
        self.h = 100

    def solve_equation(self):
        frame_count = 0
        res = 0
        hands = self.mp_hands.Hands(
            min_detection_confidence=0.7, min_tracking_confidence=0.7)
        hand_landmark_drawing_spec = self.mp_drawing.DrawingSpec(thickness=5, circle_radius=5)
        hand_connection_drawing_spec = self.mp_drawing.DrawingSpec(thickness=10, circle_radius=10)
        cap = cv2.VideoCapture(0)
        pts = deque(maxlen=512)
        # blackboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
        blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)
        digit = np.zeros((200, 200, 3), dtype=np.uint8)
        pred_class = 0
        break_taken = False
        res_list = deque(maxlen=512)
        while cap.isOpened():
            idx_to_coordinates = {}
            ret, image = cap.read()
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results_hand = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.rectangle(image, (self.x, self.y), (self.x + self.w, self.y + self.h), (0, 255, 0), 1)
            if results_hand.multi_hand_landmarks:
                for hand_landmarks in results_hand.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=hand_landmark_drawing_spec,
                        connection_drawing_spec=hand_connection_drawing_spec)
                    idx_to_coordinates = get_idx_to_coordinates(image, results_hand)
            if 8 in idx_to_coordinates and 17 in idx_to_coordinates and idx_to_coordinates[17][0] > \
                    idx_to_coordinates[8][
                        0]:
                frame_count = 0
                pts.appendleft(idx_to_coordinates[8])  # Index Finger
            if break_taken == True and len(pts) > 0:
                pts.appendleft(-1)
                pts.appendleft(pts[0])
                break_taken = False
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None or pts[i] == -1 or pts[i - 1] == -1:
                    continue
                cv2.line(image, pts[i - 1], pts[i], (0, 255, 0), 7)
                cv2.line(blackboard, pts[i - 1], pts[i], (255, 255, 255), 7)

            if 8 not in idx_to_coordinates or 17 not in idx_to_coordinates or idx_to_coordinates[17][0] <= \
                    idx_to_coordinates[8][0]:
                frame_count += 1
                break_taken = True
                if len(pts) != [] and frame_count >= 15:
                    break_taken = False
                    blackboard_gray = cv2.cvtColor(blackboard, cv2.COLOR_RGB2GRAY)
                    blur1 = cv2.medianBlur(blackboard_gray, 15)
                    blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                    thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    thresh1 = thresh1[self.y: self.y + self.h, self.x: self.x + self.w]
                    blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if blackboard_cnts is None:
                        frame_count = 0
                        continue
                    if len(blackboard_cnts) >= 1:
                        res_list = []
                        for cnt in blackboard_cnts:
                            if cv2.contourArea(cnt) > 1000:
                                x, y, w, h = cv2.boundingRect(cnt)
                                digit = blackboard_gray[y:y + h, x:x + w]
                                pred_probab, pred_class = self.math_model.predict(digit)
                                res_list.append(pred_class)
                                frame_count = 0
                    pts = deque(maxlen=512)
                    # blackboard = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    blackboard = np.zeros((720, 1280, 3), dtype=np.uint8)
            pos = 0
            if len(res_list) > 0:
                res = solve_eqn(res_list)
                res_list.clear()
            image = cv2.putText(image, str(res), (920, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            image = cv2.putText(image, "Answer =  ", (600, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("Res", rescale_frame(image, percent=100))
            cv2.imshow("BB", rescale_frame(blackboard, percent=100))

            if cv2.waitKey(5) & 0xFF == 27:
                break
        hands.close()
        cap.release()
