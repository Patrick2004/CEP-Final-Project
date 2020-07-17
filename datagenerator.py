import cv2
import numpy as np
import os

# Create the directory structure
if not os.path.exists("data"):
    os.makedirs("data")

class train:

    def main(frame):
        count = []
        train_directory = 'data/train'
        for i in os.listdir(train_directory + "/"):
            count.append(len(os.listdir(train_directory+ "/" + str(i))))

        x1 = int(0.5*frame.shape[1])
        y1 = 10
        x2 = frame.shape[1]-10
        y2 = int(0.5*frame.shape[1])
        cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,1)

        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (64, 64))

        blur = cv2.GaussianBlur(roi,(5,5),0)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

        return count, roi
