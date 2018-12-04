# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html
# https://stackoverflow.com/questions/29415719/how-do-i-create-keypoints-to-compute-sift
# https://kushalvyas.github.io/BOV.html
# https://datascience.stackexchange.com/questions/16700/confused-about-how-to-apply-kmeans-on-my-a-dataset-with-features-extracted
import cv2
import numpy as np
import os
import pandas
from sklearn.cluster import KMeans
import time
import pickle
import xlsxwriter


def file_traverse(path):
    sift = cv2.xfeatures2d.SIFT_create()
    df = pandas.DataFrame(columns=['name', 'class', 'istest', 'keypoint_count', 'features'])
    idx = 0
    for root, dirs, files in os.walk(path):
        for file_ in files:
            img = cv2.imread(os.path.join(root, file_))
            (kps, gray_img) = harris(img)
            features = sift.compute(gray_img, kps)
            df.loc[idx] = [file_,root.split('/')[1].split('\\')[0],root.split('/')[1].split('\\')[1] == 'test',len(kps), features]

            if(idx%50==0):
                print(idx)

            idx += 1
    return df

def harris(img):
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img1 = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img1, 2, 3, 0.21)
    # change last param between (0.04-0.22)
    result_img = img.copy() # deep copy image

    # Threshold for an optimal value, it may vary depending on the image.
    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    # for each dst larger than threshold, make a keypoint out of it
    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

    return (keypoints, gray_img)

def train_kmeans(feature_np,k):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(feature_np)
    filename = str(k) + 'kmodel.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    return kmeans

def load_model(k):
    filename = str(k) + 'kmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model


start_time = time.time()
image_df = file_traverse('dataset/')
feature_list = []

for index, row in image_df.iterrows():
    for feature in row['features'][1]:
        feature_list.append(feature)
print(len(feature_list))
feature_np = np.array(feature_list)


model = train_kmeans(feature_np,50)

y_kmeans = model.predict(feature_np)


print("--- %s seconds ---" % (time.time() - start_time))

# image_df['ymeans'] = y_kmeans
#
# writer = pandas.ExcelWriter('images.xlsx', engine='xlsxwriter')
# image_df.to_excel(writer, sheet_name='Sheet1')
# writer.save()


# img = cv2.imread("dataset/camera/test/image_0027.jpg")
# sift = cv2.xfeatures2d.SIFT_create()
# (kps, gray_img) = harris(img)
# features = sift.compute(gray_img, kps)
# print(len(kps))
# print(features[1][2])

# cv2.imshow('asd',gray_img)
#
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()


