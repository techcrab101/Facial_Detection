import cv2
import numpy as np

import argparse
import os


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--path", required=True,
	help="path to save image")
ap.add_argument("-a", "--vpath", required=True,
	help="path to open video")
args = vars(ap.parse_args())

path = args["path"]
vid_path = args["vpath"]
imgCount = len(os.listdir(path))


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
