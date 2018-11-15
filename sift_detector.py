import cv2
import numpy as np

img = cv2.imread("grid.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)

sift = cv2.xfeatures2d.SIFT_create()

kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp)
cv2.imwrite('sift_keypoints.jpg',img)