import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA

# Load the data
data = pd.read_excel(r'D:\srinidhi\amrita\MFC\SMOTE_analysis.xlsx')

# Convert DataFrame to NumPy arrays
X = data.iloc[:, :-1].values  # Features (embeddings)

# Reduce dimensionality using PCA for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Initialize Spectral Clustering
n_clusters = 6  # Number of clusters
spectral_clustering = SpectralClustering(n_clusters=n_clusters, random_state=42)

# Fit Spectral Clustering to the data and get cluster labels
cluster_labels = spectral_clustering.fit_predict(X)

# Calculate cluster centroids
centroids = []
for cluster in range(n_clusters):
    cluster_points = X[cluster_labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    centroids.append(centroid)

centroids = np.array(centroids)

# Visualize clusters and centroids
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster
for cluster in range(n_clusters):
    plt.scatter(X_pca[cluster_labels == cluster, 0], X_pca[cluster_labels == cluster, 1], label=f'Cluster {cluster + 1}')

# Plot centroids
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='red', label='Centroids')

plt.title('Spectral Clustering with Centroids')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
