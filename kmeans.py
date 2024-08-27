import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the data
data = pd.read_excel(r'D:\srinidhi\amrita\MFC\SMOTE_analysis.xlsx')

# Convert DataFrame to NumPy arrays
X = data.iloc[:, :-1].values  # Features (embeddings)

# Reduce dimensionality using PCA for visualization purposes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Initialize KMeans clustering
n_clusters = 6  # Number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit KMeans to the data
kmeans.fit(X)

# Get cluster labels
cluster_labels = kmeans.labels_

# Visualize clusters
plt.figure(figsize=(10, 6))

# Scatter plot for each cluster
for cluster in range(n_clusters):
    plt.scatter(X_pca[cluster_labels == cluster, 0], X_pca[cluster_labels == cluster, 1], label=f'Cluster {cluster + 1}')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='k', label='Centroids')

plt.title('KMeans Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.grid(True)
plt.show()
