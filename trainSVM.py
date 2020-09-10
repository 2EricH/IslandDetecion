# Importing the necessary modules:

from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage.io import imread
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from skimage import color
from imutils.object_detection import non_max_suppression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import imutils
import numpy as np
import argparse
import cv2
import os
import glob
from PIL import Image 
from numpy import *
import pickle

# define parameters of HOG feature extraction
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
threshold = .3


# define path to images:

pos1_im_path = "trainingData/positives1" # path for positive input dataset - individual islands
pos2_im_path = "trainingData/positives2" # path for positive input dataset - groups of islands
neg_im_path = "trainingData/negatives" # path for negative input dataset


# read the image files:
pos1_im_listing = os.listdir(pos1_im_path) # read all files in the positive image path
pos2_im_listing = os.listdir(pos2_im_path) # read all files in the positive image path
neg_im_listing = os.listdir(neg_im_path)
num_pos1_samples = size(pos1_im_listing) # total no. of images
num_pos2_samples = size(pos2_im_listing) # total no. of images
num_neg_samples = size(neg_im_listing)

print("Positive samples class 1: ", num_pos1_samples) 
print("Positive samples class 2: ", num_pos2_samples) 
print("Negative samples: ", num_neg_samples)

data= []
labels = []

# compute HOG features and label them:

for file in pos1_im_listing: 
    img = Image.open(pos1_im_path + '\\' + file) 
    img = img.resize((64,128))
    gray = img.convert('L') # convert the image into single channel
    # HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    data.append(fd)
    labels.append(1)
    # print("Data Shapweee test: ", len(data))

for file in pos2_im_listing: 
    img = Image.open(pos2_im_path + '\\' + file) 
    img = img.resize((64,128))
    gray = img.convert('L') # convert the image into single channel
    # HOG for positive features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True)# fd= feature descriptor
    data.append(fd)
    labels.append(2)
    
# Same for the negative images
for file in neg_im_listing:
    img= Image.open(neg_im_path + '\\' + file)
    img = img.resize((64,128))
    gray= img.convert('L')
    # HOG for negative features
    fd = hog(gray, orientations, pixels_per_cell, cells_per_block, block_norm='L2', feature_vector=True) 
    data.append(fd)
    labels.append(0)
# encode the labels
le = LabelEncoder()
labels = le.fit_transform(labels)


print("Constructing training/testing split...")
print("Finding optimal C and Gama values - this can take a few minutes...\n")

data = np.array(data, ndmin=2)
# print(data.shape)


param_grid = [{'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear']},
              {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['rbf']},
              {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['poly']}]

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
(X_train, X_test, Y_train, Y_test) = train_test_split(data, labels)
grid_search.fit(X_train, Y_train)
grid_search.score(X_test, Y_test)
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_*100, "%")
print(grid_search.best_estimator_)


print("------------------------------------------")

print("Training Linear SVM classifier...")
# svm = LinearSVC(C=best_C, gamma=best_gamma)
svm = grid_search.best_estimator_
svm.fit(X_train, Y_train)

print(" Evaluating classifier on test data ...")
predictions = svm.predict(X_test)
print(classification_report(Y_test, predictions))

# print("Prediction: ", predictions)
# print("    Actual: ", Y_test)
print("------------------------------------------")


# Save the svm:
filename = 'predictor.sav'
pickle.dump(svm, open(filename, 'wb'))
