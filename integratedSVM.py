import numpy as np
import sys
import cv2
import os
import shutil
from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
import imutils
import argparse
import glob
from PIL import Image
import time
from numpy import *
import pickle


config = 'predictor.sav'
imageStore = os.listdir("./imageStore")
svm = pickle.load(open(config, 'rb'))




def preProcessLocation(imPath): 
    #Loading of origional image

    rawImage = cv2.imread(imPath)
    brightImage = rawImage.copy()
    rawImage = cv2.fastNlMeansDenoisingColored(rawImage,None,10,10,7,21)
    gray = cv2.cvtColor(brightImage, cv2.COLOR_BGR2GRAY)
    
    # Uncomment to see origional Immage

    # cv2.imshow("origional", rawImage)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()
    #----------------------------------------
    # Find area of the image with the largest intensity value

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
    maxLoc = np.transpose(maxLoc)
    maxLoc = tuple([maxLoc[1], maxLoc[0]])

    print("Location of brightest pixel: ", maxLoc)
    # cv2.circle(brightImage, maxLoc, 1, (255, 0, 0), 2)

    print("Intensity: ", gray[maxLoc])


    # display where the brightest area of pixels is

#     cv2.imshow("Brightest Spot", brightImage)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    #----------------------------------------
    # Preprocessing (using a bilateral filter)

    height = int(rawImage.shape[0])
    width = int(rawImage.shape[1])
    blackImage = np.zeros((height,width,3), np.uint8)
   
    bilateral_filtered_image = cv2.bilateralFilter(rawImage, 5, height, width)
    # cv2.imshow('Bilateral', bilateral_filtered_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #----------------------------------------
    # Edge detection
    # 75 and 200 default min/max for canny edge detection


    print("Median: ", np.median(bilateral_filtered_image))
    edge_detected_image = auto_canny(bilateral_filtered_image)
    print("Detecting islands.....")


    # if gray[maxLoc] <= 50:
    #     edge_detected_image = cv2.Canny(bilateral_filtered_image, 0, 10) # For dark images 
    # else:    
    #     edge_detected_image = cv2.Canny(bilateral_filtered_image, 10, 100) # For bright images

    cv2.imshow('Edge detected image', edge_detected_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
#     shutil.rmtree('./Temp Images')
#     os.mkdir("./Temp Images")
    #----------------------------------------
    # Finding Contours
   
    _, contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# print(contours)
    

    contour_list = []
    for contour in contours:        
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
    #     if area > 0:
    #         print("Pixel area: ", area)
        if ((len(approx) > 0) or (area > 0) ):  # len 8, area 30 are default
            contour_list.append(contour)
    #----------------------------------------
    # convert the grayscale image to binary image
    # ret,thresh = cv2.threshold(gray,127,255,0)

    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
    lowThresh = 0.5*ret

    # calculate moments of binary image
    os.chdir("./imageStore")
    imCount = 1
    for cnt in contour_list:
        M = cv2.moments(cnt)
        if int(M["m00"] != 0):
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
#             cv2.circle(rawImage, (cX, cY), 2, (0, 0, 255), 1)   # Draw centers in relation to moments
            
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(rawImage,(x,y),(x+w,y+h),(0,255,0),2)

            if (y-1 == -1) or (x-1 == -1):	
            	cropped = edge_detected_image[y:y+h+1, x:x+w+1]
            	fileName = "edgeImage" + str(imCount) + ".png"
            	# print(y, x)
            else:
            	cropped = edge_detected_image[y-1:y+h+1, x-1:x+w+1]
            	fileName = "croppedImage" + str(imCount) + ".png"
#             print(fileName)
            cv2.imwrite(fileName, cropped)
            imCount += 1

            # cv2.imshow('Cropped island hits', cropped)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            if classifyObject(fileName) == [1]:
            	suppPath = './suppTrain/hits'
            	cv2.imwrite(os.path.join(suppPath, fileName), cropped)
            	cv2.rectangle(brightImage, (x,y), (x+w,y+h),(0,255,0),2)
            else: 
            	suppPath = './suppTrain/negs'
            	cv2.imwrite(os.path.join(suppPath, fileName), cropped)

    # cv2.imshow("Rects", rawImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
         
           

    #----------------------------------------

    # Displaying Resuts
    # cv2.imshow('Library Detected Image',rawImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imshow('Objects Detected',brightImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preProcessFeatures(islandHit):
	# if os.getcwd())
	orientations = 9
	pixels_per_cell = (8, 8)
	cells_per_block = (2, 2)
	threshold = .3
	try:
		img = Image.open(islandHit)
	except OSError:
		print("Bad file: ", islandHit)
		return np.array([[1], [1]])

	img = img.resize((64,128))
	gray = img.convert('L') 
	# HOG for positive features
	fd, hog_image = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, visualize=True)


	# cv2.imshow("hogFilter", hog_image)
	# cv2.waitKey(0) 
	# cv2.destroyAllWindows()

	X_new = np.array(fd, ndmin=2)
	return X_new


def classifyObject(image):

	X_new = preProcessFeatures(image)
	if X_new.shape[1] == 1:
		return -1
	return(svm.predict(X_new))

def auto_canny(image):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	if 0 <= v <= 10:
		sigma = 10
	if 10 < v <= 12:
		sigma = 8
	if 12 < v <= 15:
		sigma = 5
	if 15 < v:
		sigma = 0.22
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged
     

shutil.rmtree('./imageStore')
time.sleep(.0001)
os.mkdir("./imageStore")

imagePath = "./images/chainformation4.png" # Median 42, best sigma = 0.22
# imagePath = "./images/chainformation3.png" # Median 46, best sigma = 0.22
# imagePath = "./images/chainformation2.png" # Median 45, best sigma = 0.22
# imagePath = "./images/chainformation1.png" # Median 30, best sigma = 0.22
# imagePath = "./Images/islandtest3.tif"   # Median 12, best sigma 8
# imagePath = "./Images/islandtest1.tif"   # Median 15, best sigma 5
# imagePath = "./Images/islandtest2.tif"   # Median 0, best sigma n/a 
preProcessLocation(imagePath)
