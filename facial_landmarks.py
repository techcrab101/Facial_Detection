# import the necessary packages
from imutils import face_utils
import numpy as np
import imutils
from utilities import column
from utilities import centeroidnp
import dlib
import cv2
import math
import constants

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(constants.shapePredictor)

def get_landmarks(rect, gray):
	# determine the facial landmarks for the face region, then
	# convert the facial landmark (x, y)-coordinates to a NumPy
	# array
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	
	# convert dlib's rectangle to a OpenCV-style bounding box
	# [i.e., (x, y, w, h)], then draw the face bounding box
	# (x, y, w, h) = face_utils.rect_to_bb(rect)
	
	centerX, centerY = centeroidnp(shape)
	#cv2.circle(image, (centerX, centerY), 1, (255, 0, 0), -1)

	# show the face number
	# cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
	#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

	# loop over the (x, y) coordinates for the facial landmarks
	# and draw them on the image
	landmarks = []
	landmarks.append((centerX, centerY))
	for (x, y) in shape:
		landmarks.append((x,y))

		
	return landmarks 

def get_normalized_landmarks(landmarks):
	xlist = column(landmarks, 0)
	ylist = column(landmarks, 1)
	xlist.pop(0)
	ylist.pop(0)
	
	xmean = landmarks[0][0]
	ymean = landmarks[0][1]
	
	xcom = xlist[31] - xlist[28]
	ycom = ylist[31] - ylist[28]

	angle = 0

	if xcom == 0:
		angle = 90
	else:
		angle = (math.atan(abs(ycom)/abs(xcom)) * 180) / (math.pi)

	if xcom > 0:
		angle = 180 - angle

	dist = math.sqrt((xcom)**2 + (ycom)**2)
	
	dist_multiplier = 1.0 / dist

	landmarks_vectorized = []

	for x, y in zip(xlist, ylist):
		xdiff = x - xmean
		ydiff = ymean - y

		# if xdiff negative than on west hemisphere
		# if ydiff negative then on southern hemisphere
		
		dist_coord = math.sqrt((xdiff)**2 + (ydiff)**2)

		dist_coord = dist_coord * dist_multiplier

		angle_coord = 0

		if xdiff == 0:
			angle_coord = 90 * (-1 if ydiff < 0 else 1)
		else:
			angle_coord = (math.atan(abs(ydiff)/abs(xdiff)) * 180) / (math.pi)

		if xdiff > 0 and ydiff < 0:
			angle_coord = 360 - angle_coord
		if xdiff < 0 and ydiff < 0:
			angle_coord = angle_coord + 180
		if xdiff < 0 and ydiff > 0:
			angle_coord = 180 - angle_coord

		angle_coord = angle_coord - angle

		landmarks_vectorized.append(dist_coord)
		landmarks_vectorized.append(angle_coord)

	if landmarks_vectorized == []:
		return None

	return landmarks_vectorized

