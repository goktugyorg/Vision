from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import pandas
import pickle

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=11)
model = kmeans.fit(X)
y_kmeans = kmeans.predict(X)

filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
y = loaded_model.predict(X)
print(len(X))
print(y)

# buckets = [0] * 100

# df = pandas.DataFrame(columns=['name', 'class', 'istest', 'keypoint_count', 'features'])
#
# df['ymeans'] = y_kmeans

# print(df)