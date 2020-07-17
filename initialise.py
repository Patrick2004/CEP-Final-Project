import numpy as np
from keras.models import model_from_json
from operator import xor
import cv2, sys, os, argparse
from collections import deque

def callback(value):
   pass

def setup_trackbars(range_filter):
   cv2.namedWindow("Trackbars", 0)

   for i in ["MIN", "MAX"]:
       v = 0 if i == "MIN" else 255
       for j in range_filter:
           cv2.createTrackbar("%s_%s" % (j, i), "Trackbars", v, 255, callback)

def get_trackbar_values(range_filter):
   values = []
   for i in ["MIN", "MAX"]:
       for j in range_filter:
           v = cv2.getTrackbarPos("%s_%s" % (j, i), "Trackbars")
           values.append(v)

   return values

class INITIALISE:

    def __init__(self):
       self.range_filter = "HSV".upper()
       self.state = True

    def main(self, frame):
        if self.state == True:
            setup_trackbars(self.range_filter)
            self.state = False
        else:
            pass

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        v1_min, v2_min, v3_min, v1_max, v2_max, v3_max = get_trackbar_values(self.range_filter)

        frame = cv2.GaussianBlur(frame, (5, 5), cv2.BORDER_DEFAULT)
        thresh = cv2.inRange(frame, (v1_min, v2_min, v3_min), (v1_max, v2_max, v3_max))

        return thresh

class GESTURE_PREDICTION():

    def __init__(self, model, weights):
        json_file = open(model, "r")
        model_json = json_file.read()
        json_file.close()

        self.model = model_from_json(model_json)
        self.model.load_weights(weights)

    def classify(self, x, y, w, h, frame):
        l = max(w, h) + 300
        roi = frame[y-300: y+l, x-300: x+l]
        roi = roi.astype(np.uint8)

        try:
            test_image = cv2.resize(roi, (64, 64), cv2.INTER_AREA)

            prediction = self.model.predict(test_image.reshape(1, 64, 64, 1))
            result_class = prediction.argmax(axis=1)[0]

            return  result_class, test_image

        except:
            pass
