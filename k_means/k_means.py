import numpy as np
import matplotlib.pyplot as plt


class KMeans(object):
    def __init__(self, n_clusters=3, max_iter=50,plot_step=False):
        self.K = n_clusters
        self.max_iter = max_iter
        self.plot_step = plot_step
        self.clusters = [[]for _ in range(self.K)]
        self.centroids = []

    def fit(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = X[random_sample_idxs]


        for _ in range(self.max_iter):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_step:
                self.plot()

            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            if self._is_converged(centroids_old, self.centroids):
                break

    def predict(self, X):
        labels = [self._closest_centroid(sample, self.centroids) for sample in X]
        return np.array(labels)

    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            cent_idx = self._closest_centroid(sample, centroids)
            clusters[cent_idx].append(idx)

        return clusters

    def _closest_centroid(self, sample, centroids):
        distances = [self._euclidean(sample,cent) for cent in centroids]
        idx = np.argmin(distances)
        return idx

    def _euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids


    def _is_converged(self, centroids_old, centroids):
        dist =[self._euclidean(centroids_old[i], centroids[i]) for i in range(len(centroids))]
        return sum(dist) < 1e-6

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 6))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)

        plt.show()


from sklearn.datasets import make_blobs
np.random.seed(42)

X, y = make_blobs(centers=3, n_samples=300, n_features=2, shuffle=True, random_state=42)
# Shuffling effects which samples are chosen as initial centroids.


km = KMeans(n_clusters=3, max_iter=150, plot_step=True)
km.fit(X)