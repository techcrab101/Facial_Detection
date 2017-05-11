# import the necessary packages
from sklearn.externals import joblib
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import math
import facial_landmarks

import datetime
import constants
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--path', required=True,
		help='path to classifier')
args = vars(ap.parse_args())

cap = cv2.VideoCapture(constants.cam)

clf = joblib.load(args['path'])
definitions = constants.classifierDefinitions
detector = facial_landmarks.detector


while True:
	_, image = cap.read()
	image = imutils.resize(image, width=constants.imgWidth)
	image_mesh = image.copy()
	gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

	rects = detector(gray, 1)


	for (face_count, rect) in enumerate(rects):
		landmarks = facial_landmarks.get_landmarks(rect, gray)
		for (x,y) in landmarks:
			cv2.circle(image_mesh, (x,y), 1, (0,0,255), -1)
			cv2.line(image_mesh, landmarks[0], (x,y), (255,0,0), 1)
			pass
		
		landmarks_vec = facial_landmarks.get_normalized_landmarks(landmarks)
		if landmarks_vec == None:
			break
		landmarks_vec = np.array([landmarks_vec]).reshape(1,-1)
		
		prediction = clf.predict(landmarks_vec)
		prediction_prob = clf.predict_proba(landmarks_vec)
	
		if np.amax(prediction_prob) < 0.75:
			break

		#print (definitions[int(prediction[0])],
		#		prediction_prob[0][prediction[0]])
		print ('prediction:' , prediction)
		print ('prediction prob:', prediction_prob)
		cv2.putText(image, definitions[int(prediction[0])],
				(landmarks[0][0] - 30, landmarks[0][1] - 50), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# show the output image with the face detections + facial landmarks
	cv2.imshow("image", cv2.resize(image, (0,0),
			fx=constants.imgResizeMultiplier,
			fy=constants.imgResizeMultiplier))
	cv2.imshow('mesh', image_mesh)
	key = cv2.waitKey(1) & 0xFF
	if key == ord('q') or key == 27:
		break;

cv2.destroyAllWindows()
