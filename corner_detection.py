# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
# https://stackoverflow.com/questions/29415719/how-do-i-create-keypoints-to-compute-sift
# https://kushalvyas.github.io/BOV.html
# https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted
import cv2
import numpy as np

def harris(img):
    '''
    Harris detector
    :param img: an color image
    :return: keypoint, image with feature marked corner
    '''

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img1 = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img1, 2, 3, 0.15)
    # change last param between (0.04-0.18)
    result_img = img.copy() # deep copy image

    # Threshold for an optimal value, it may vary depending on the image.
    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # for each dst larger than threshold, make a keypoint out of it
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

    return (keypoints, gray_img)

sift = cv2.xfeatures2d.SIFT_create()
img = cv2.imread("image_0003.jpg")

(kps, gray_img) = harris(img)
features = sift.compute(gray_img, kps)

print(len(kps))
print(features[1][0])

# cv2.imshow('asd',gray_img)
#
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


