import pandas as pd
import numpy as np
import sys
import os
from collections import Counter

if len(sys.argv) == 5:
    input_file = sys.argv[1]
    n_clusters = int(sys.argv[2])
    eps = int(sys.argv[3])
    minPts = int(sys.argv[4])

else:
    print("Usage: clustering.py <input_file> <n_clusters> <eps> <minPts>")

print('Done: valid command input')

# input_file = "Assignment_3_files\Assignment 3 input data\input1.txt"
# n_clusters = 8
# eps = 15
# minPts = 22

# Load the data from a text file
data = pd.read_csv(input_file, sep='\t', header=None, names=['object_id', 'x', 'y'])

# Convert the data into a 2D numpy array
X = data[['x', 'y']].values

# Define Euclidean distance function
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Define range query function
def range_query(DB, distFunc, Q, eps):
    neighbors = []
    for P in range(len(DB)):
        if distFunc(DB[P], DB[Q]) < eps:
            neighbors.append(P)
    return neighbors

# Define DBSCAN function
def dbscan(DB, distFunc, eps, minPts):
    C = 0
    labels = [0]*len(DB)

    for P in range(len(DB)):
        if P % 500 == 0:
            print("seed no.", P)
        if not (labels[P] == 0):
            continue
        neighbors = range_query(DB, distFunc, P, eps)
        if len(neighbors) < minPts:
            labels[P] = -1
            continue
        C += 1
        labels[P] = C
        seed_set = [n for n in neighbors if n != P]
        for Q in seed_set:
            if labels[Q] == -1:
                labels[Q] = C
            if labels[Q] != 0:
                continue
            labels[Q] = C
            neighbors = range_query(DB, distFunc, Q, eps)
            if len(neighbors) >= minPts:
                seed_set += neighbors
    return labels

labels = dbscan(X, euclidean_distance, eps, minPts)
print('Done Labeling')

data['cluster'] = labels
data['object_id'] = data['object_id'].astype(str)
# data.to_csv(r"C:\Users\DeRoxy\Documents\GitHub\HYU-DataScience-Clustering-Classification-by-using-DBSCAN\Assignment_3_files\Assignment 3 input data\output1.csv", index=False)
# print('Done export data')

# # Count number of points in each cluster
# cluster_counts = Counter(labels)
# print('Done cluster_counts: ', cluster_counts)

# If there are more than n_clusters, remove the smallest ones
# if len(cluster_counts) > n_clusters:
#     smallest_clusters = [cluster for cluster, count in cluster_counts.most_common()[:-n_clusters-1:-1]]
#     labels = [label if label not in smallest_clusters else -1 for label in labels]

# Create output directory if it doesn't exist
output_dir = os.path.splitext(input_file)[0]
os.makedirs(output_dir, exist_ok=True)
print('Done output_dir: ', output_dir)

# Output each cluster to a separate file
for cluster in set(labels):
    if cluster == -1:  # Skip noise points
        continue
    with open(f"{output_dir}_cluster_{cluster-1}.txt", 'w') as f:
        for i, label in enumerate(labels):
            if label == cluster:
                f.write(f"{data.iloc[i]['object_id']}\n")
    print('Done separate file: ', output_dir, cluster-1)