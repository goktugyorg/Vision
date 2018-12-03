from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import pandas

X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=0)

kmeans = KMeans(n_clusters=11)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

print(len(X))
print(y_kmeans)

df = pandas.DataFrame(columns=['name', 'class', 'istest', 'keypoint_count', 'features'])

df['ymeans'] = y_kmeans

print(df)