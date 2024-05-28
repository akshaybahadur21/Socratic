import cv2
import mediapipe as mp


class HandDetector:
    def __init__(self, mode=False, max_hands=2, detect_confidence=0.5, track_confidence=0.5):
        self.track_confidence = track_confidence
        self.detect_confidence = detect_confidence
        self.max_hands = max_hands
        self.mode = mode

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.mode, self.max_hands, 1, self.detect_confidence, self.track_confidence)


        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, image, draw=True):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img)
        if self.results.multi_hand_landmarks:
            for hands in self.results.multi_hand_landmarks:
                if draw:
                    self.mp_draw.draw_landmarks(img, hands, self.mp_hands.HAND_CONNECTIONS)

        return img

    def find_position(self, image, hand_no=0, draw=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            curr_hand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(curr_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.landmark_list.append([id, cx, cy])
                if draw:
                    cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return self.landmark_list

    def fingers(self):
        finger_list = []
        if self.landmark_list[self.tip_ids[0]][1] < self.landmark_list[self.tip_ids[0] - 1][1]:
            finger_list.append(1)
        else:
            finger_list.append(0)

        for id in range(1, 5):
            if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                finger_list.append(1)
            else:
                finger_list.append(0)

        return finger_list
