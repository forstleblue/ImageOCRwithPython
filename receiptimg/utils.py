import numpy as np
import sys
import cv2
import math
import os
import cPickle
opencv3 = cv2.__version__[0] == '3'

# 1 do thresholding, 0 otherwise
thresh=1

# Initialize sift detector
if opencv3:
	sift = cv2.xfeatures2d.SIFT_create()
else:
	sift = cv2.SIFT()

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)

# Initialize flann matcher
flann = cv2.FlannBasedMatcher(index_params,search_params)


# Initialize SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()
 
params.minRepeatability = 2;
params.minDistBetweenBlobs = 10;

# Change thresholds
params.minThreshold = 1;
params.maxThreshold = 420;
 
# Filter by Area.
params.filterByArea = True
params.minArea = 0.000001
params.maxArea = 5000
 
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.8
params.maxCircularity = 9999999999999
 
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.95
params.maxConvexity = 999999999999999999999
 
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0
params.maxInertiaRatio = 0.1

# create blob detector with parameters

#detector = cv2.SimpleBlobDetector_create(params)
detector = cv2.SimpleBlobDetector(params)

# Minimal matches to be found
MIN_MATCH_COUNT = 15
MIN_MATCH_COUNTLoto = 5
MIN_MATCH_COUNTJeu = 5

def rotateImage(image, angle):
	row,col = image.shape
	center=tuple(np.array([row,col])/2)
	rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
	new_image = cv2.warpAffine(image, rot_mat, (col,row))
	return new_image

def deleteBlobs(image):
	w, h = image.shape
	# Detect blobs
	keypoints = detector.detect(image)
	for keypoint in keypoints:
		if keypoint.size < 12:
			cv2.circle(image, (int(keypoint.pt[0]),int(keypoint.pt[1])), int(keypoint.size), (255, 255, 255), -1)
	return image

def read_desc_from_file(filename):
    package_directory = os.path.dirname(os.path.abspath(__file__))
    if os.path.getsize(os.path.join(package_directory, filename)) <= 0:
        return np.array([])
    f = np.load(os.path.join(package_directory, filename))
    if f.size == 0:
        return np.array([])
    f = np.atleast_2d(f)
    return f # feature locations, descriptors

def write_desc_to_file(filename, desc):
    np.save(filename, desc)
	
def read_keyps_from_file(filename):
	package_directory = os.path.dirname(os.path.abspath(__file__))
	index = cPickle.loads(open(os.path.join(package_directory, filename)).read())
	kp1 = []
	for point in index:
		temp = cv2.KeyPoint(x=point[0][0],y=point[0][1],_size=point[1], _angle=point[2], _response=point[3], _octave=point[4], _class_id=point[5])
		kp1.append(temp)
	return kp1

def write_keyps_to_file(filename, keyps):
	index = []
	for point in keyps:
		temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
		index.append(temp)
	# Dump the keypoints
	f = open(filename, "w")
	f.write(cPickle.dumps(index))
	f.close()
