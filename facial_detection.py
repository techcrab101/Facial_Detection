# import the necessary packages
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import datetime
import facial_landmarks
import constants

cap = cv2.VideoCapture(constants.cam)

detector = facial_landmarks.detector

while True:
	_, image = cap.read()
	image = imutils.resize(image, width=constants.imgWidth)
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)


	for (face_count, rect) in enumerate(rects):
		landmarks = facial_landmarks.get_landmarks(rect, gray)
		for (x,y) in landmarks:
			cv2.circle(image, (x,y), 1, (0,0,255), -1)
			cv2.line(image, landmarks[0], (x,y), (255,0,0), 1)
			pass

	# show the output image with the face detections + facial landmarks
	cv2.imshow("image", cv2.resize(image, (0,0),
			fx=constants.imgResizeMultiplier,
			fy=constants.imgResizeMultiplier))
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q') or key == 27:
		break;

cv2.destroyAllWindows()
