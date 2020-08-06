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
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image
from numpy import *
import pickle

filename = 'predictor.sav'
svm = pickle.load(open(filename, 'rb'))
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3

def preProcess(imPath): 

    rawImage = cv2.imread(imPath)
    brightImage = rawImage.copy()
    rawImage = cv2.fastNlMeansDenoisingColored(rawImage,None,10,10,7,21)
    gray = cv2.cvtColor(brightImage, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    # cv2.imshow("origional", rawImage)
    # cv2.waitKey(0) 
    # cv2.destroyAllWindows()

    #----------------------------------------
    # Find area of the image with the largest intensity value

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)

    # print(maxLoc)
    print("Intensity: ", gray[maxLoc])

    #----------------------------------------
    # Preprocessing (using a bilateral filter)

    height = int(rawImage.shape[0])
    width = int(rawImage.shape[1])
    blackImage = np.zeros((height,width,3), np.uint8)
   
    bilateral_filtered_image = cv2.bilateralFilter(rawImage, 5, height, width)

    #----------------------------------------
    # Edge detection
    # 75 and 200 default min/max canny

    if gray[maxLoc] <= 30:
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 0, 10) # For dark images 
    else:    
        edge_detected_image = cv2.Canny(bilateral_filtered_image, 10, 100) # For bright images

 
    # Saving Results     
    os.chdir("./tempImages")
    fileName = "passable.png"
    cv2.imwrite(fileName, edge_detected_image)
    os.chdir("..")
    
imagePath = "./trainingData/tests/test2.png"
preProcess(imagePath)

img = Image.open("./tempImages/passable.png") 
img = img.resize((64,128))
gray = img.convert('L') 
# HOG for positive features
fd, hog_image = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True, visualize=True)

# cv2.imshow("hogFilter", hog_image)
# cv2.waitKey(0) 
# cv2.destroyAllWindows()

X_new = np.array(fd, ndmin=2)
# print(X_new.shape)


print(svm.predict(X_new))
if svm.predict(X_new) == 0:
	print("No island")
else:
	print("Island")