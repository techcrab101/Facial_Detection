# import the necessary packages
#from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib

from imutils import face_utils

import numpy as np
import imutils
import dlib
import cv2
import facial_landmarks

import math
import datetime
import os
from utilities import printProgressBar
import constants
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True, 
	help="path to save classifier")
args = vars(ap.parse_args())

clf_path = args['path']

features = []
labels = []

detector = facial_landmarks.detector

path = 'Data/'



print ('Loading Data...')

t0 = datetime.datetime.now()

i = 0
imageCount = 0
for dir in os.listdir(path):
	for filename in os.listdir(path + dir):
		imageCount += 1

if imageCount == 0:
	raise Exception('No Image Data found')

printProgressBar(i , imageCount, prefix = 'Progress:', suffix = 'Complete')

for dir in os.listdir(path):
	for filename in os.listdir(path + dir):
		image = cv2.imread(path+dir+'/'+filename)
		image = imutils.resize(image, width=constants.imgWidth)
		gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

		rects = detector(gray, 1)
		
		i += 1
		printProgressBar(i, imageCount, prefix = 'Progress:', suffix = 'Complete')
		for (face_count, rect) in enumerate(rects):
			landmarks = facial_landmarks.get_landmarks(rect, gray)
			for (x,y) in landmarks:
				cv2.circle(image, (x,y), 1, (0,0,255), -1)
				cv2.line(image, landmarks[0], (x,y), (255,0,0), 1)
				pass
			landmarks_vec = facial_landmarks.get_normalized_landmarks(landmarks)
			if not(landmarks_vec == None):
				features.append(landmarks_vec)
				labels.append(int(dir))
		
		cv2.imshow('training image', image)
		
		cv2.waitKey(1)

t1 = datetime.datetime.now()
diff = t1 - t0

print ('Data loaded')
print ('Data loading time:', (diff.days*1440)+(diff.seconds/60), 'mins')

dataSize = len(landmarks_vec)
layerCount = constants.layerCount

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, 
#			hidden_layer_sizes=(layerCount, dataSize), random_state=1)

clf = SVC(kernel='linear', probability=True, tol=1e-3, verbose=True)

t0 = datetime.datetime.now()

print("training...")
clf.fit(features, labels)
print("trained")

t1 = datetime.datetime.now()

diff = t1 - t0

print('training time:',(diff.days * 1440)+ (diff.seconds / 60), "mins")

print("Saving...")
joblib.dump(clf, clf_path)
print("Saved")

