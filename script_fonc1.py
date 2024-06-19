import sys
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


def load_centroids(centroids_file):
    """Load the centroids from a JSON file."""
    with open(centroids_file, 'r') as f:
        centroids = json.load(f)
    return np.array(centroids)


def main():
    if len(sys.argv) != 4:
        print("Usage: python unsupervised_learning.py <data_file> <centroids_file> <features>")
        sys.exit(1)

    data_file = sys.argv[1]
    centroids_file = sys.argv[2]
    features = sys.argv[3].split(',')

    # Load data and centroids
    data = load_data(data_file)
    centroids = load_centroids(centroids_file)

    # Select the specified features
    X = data[features].values

    # Create a KMeans instance with the specified centroids
    kmeans = KMeans(init=centroids, n_clusters=len(centroids), n_init=1)

    # Fit the model and predict the clusters
    kmeans.fit(X)
    clusters = kmeans.predict(X)

    # Add the cluster labels to the dataframe
    data['cluster'] = clusters

    # Convert the result to JSON
    result = data.to_json(orient='records')

    # Print the result
    print(result)


if __name__ == '__main__':
    main()
