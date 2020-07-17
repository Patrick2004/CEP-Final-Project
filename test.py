import argparse
import cv2
import pyautogui
from yolo import YOLO
from initialise import GESTURE_PREDICTION, INITIALISE

screenWidth, screenHeight = pyautogui.size()

yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
gesturePred = GESTURE_PREDICTION("models/model.json", "models/model.h5")
init = INITIALISE()

yolo.size = int(256)
yolo.confidence = float(0.2)

camera = cv2.VideoCapture(0)

class BACKEND:

    def calibrate(frame):
        hsv = init.main(frame)
        return hsv
    def predict(frame, state):
        aug = cv2.resize(frame, (screenWidth, screenHeight))
        width, height, inference_time, results = yolo.inference(aug)

        if state == False:
            thresh = init.main(aug)
        else:
            blur = cv2.GaussianBlur(aug,(5,5),0)
            roi = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        for detection in results:
            id, name, confidence, x, y, w, h = detection
            cx = x + (w / 2)
            cy = y + (h / 2)
            cz = w*h

        try:
            pred, roi = gesturePred.classify(x, y, w, h, thresh)
            return roi, pred, cx, cy

        except:
            pass
