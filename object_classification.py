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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import xlsxwriter


def file_traverse(path):
    # traverse the dataset and create the dataframe for whole dataset
    # for each image creates sift features
    sift = cv2.xfeatures2d.SIFT_create()
    df = pandas.DataFrame(columns=['name', 'class', 'istest', 'keypoint_count', 'features'])
    idx = 0
    for root, dirs, files in os.walk(path):
        for file_ in files:
            img = cv2.imread(os.path.join(root, file_))
            (kps, gray_img) = harris(img)
            features = sift.compute(gray_img, kps)
            df.loc[idx] = [file_,root.split('/')[1].split('\\')[0],root.split('/')[1].split('\\')[1] == 'test',len(kps), features]

            idx += 1
    return df

def harris(img):
    # detect corners and return keypoinrs
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray_img1 = np.float32(gray_img)
    dst = cv2.cornerHarris(gray_img1, 2, 3, 0.21)
    result_img = img.copy()

    result_img[dst > 0.01 * dst.max()] = [0, 0, 255]

    keypoints = np.argwhere(dst > 0.01 * dst.max())
    keypoints = [cv2.KeyPoint(x[1], x[0], 1) for x in keypoints]

    return (keypoints, gray_img)

def train_kmeans(feature_np,k):
    # trains and saves kmeans model
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(feature_np)
    filename = str(k) + 'kmodel.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    return kmeans

def load_model(k):
    filename = str(k) + 'kmodel.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

def save_df(df,name):
    writer = pandas.ExcelWriter(str(name) + '.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()

k = 100
start_time = time.time()

image_df = file_traverse('dataset/')
feature_list = []

for index, row in image_df.iterrows():
    for feature in row['features'][1]:
        feature_list.append(feature)

feature_np = np.array(feature_list)

model = load_model(k)
y_kmeans = model.predict(feature_np)

histograms = [[0 for x in range(k)] for y in range(len(image_df))]

# create histograms
pos = 0
for index,row in image_df.iterrows():
    for feature in row['features'][1]:
        histograms[index][int(y_kmeans[pos])] += 1
        pos += 1

# normalize histograms
for idx, val in enumerate(histograms):
    histograms[idx] = [x / len(image_df.loc[idx]['features'][1]) for x in histograms[idx]]

# knn
# create test and train data
test = []
train = []
label = []
true_values = []
for index, row in image_df.iterrows():
    if row['istest'] == False:
        train.append(histograms[index])
        label.append(row['class'])
    else:
        test.append(histograms[index])
        true_values.append(row['class'])

# classify
neigh = KNeighborsClassifier(n_neighbors=5)
neigh.fit(train, label)


pred_values = []
for tester in test:
    pred_values.append(neigh.predict([tester]))

# confusion matrix
con_mat = confusion_matrix(true_values, pred_values, list(set(true_values)))

print(con_mat)
print(classification_report(true_values, pred_values))

print("--- %s seconds ---" % (time.time() - start_time))



