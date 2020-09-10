import cv2
import numpy as np

# Load the image
img = cv2.imread("./suppTrain/groups/croppedImage90.png")
# Convert to greyscale
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Convert to binary image by thresholding
# _, threshold = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY_INV)
# Find the contours
_, contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)# print(contours)
    

contour_list = []
for cnt in contours:
    epsilon = 0.01*cv2.arcLength(cnt, True)        
    approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    area = cv2.contourArea(cnt)

    print(len(approx))


    # Position for writing text
    x,y = approx[0][0]

    if len(approx) == 3:
        cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
    elif len(approx) == 5:
        cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, 0,2)
    elif 6 < len(approx) < 15:
        cv2.putText(img, "Ellipse", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,0))
    else:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255))
cv2.imshow("final", img)
cv2.waitKey(0)