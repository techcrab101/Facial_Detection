import cv2
import numpy as np

import argparse
import os

cap = cv2.VideoCapture(0)


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to save image")
args = vars(ap.parse_args())

path = args["path"]
imgCount = len(os.listdir(path))


print ('Press C to capture image')

i = 0
while True:
	_, img = cap.read()
	
	cv2.imshow("img", img)

	key = cv2.waitKey(1) & 0xFF
	if key == ord('c'):
		cv2.imwrite(path + str(i+imgCount) +'.png', img)
		
		print (i)
		i += 1

	if key == 27 or key == ord('q'):
		break

cv2.destroyAllWindows()
