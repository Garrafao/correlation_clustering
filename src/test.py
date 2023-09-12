import networkx as nx
from itertools import combinations
from correlation import cluster_correlation_search
from utils import get_clusters, transform_edge_weights
import numpy as np

# Define true clusters
nodes = ['node1', 'node2', 'node3', 'node4']
node2clusters_true = {'node1':0, 'node2':0, 'node3':1, 'node4':1}
print('clusters_true', node2clusters_true)

# Initialize graph
graph = nx.Graph()

# Generate graph
for (u,v) in combinations(nodes, 2):
    if node2clusters_true[u] == node2clusters_true[v]:
        graph.add_edge(u, v, weight=np.random.choice([2,3,4]))
    else:
        graph.add_edge(u, v, weight=np.random.choice([1,2,3]))

# Prepare graph for clustering
threshold = 2.5
weight_transformation = lambda x: x-threshold
graph = transform_edge_weights(graph, transformation = weight_transformation) # shift edge weights

# Cluster graph
clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_attempts = 100, max_iters = 200)

# Display results
node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
node2cluster_inferred = {node:node2cluster_inferred[node] for node in nodes}
print('clusters_inferred', node2cluster_inferred)

# Clustering again and initializing with the previous solution can improve the solution in many cases (this can be done multiple times)
clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_attempts = 100, max_iters = 200, initial = clusters)

# Display results after second iteration
node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
node2cluster_inferred = {node:node2cluster_inferred[node] for node in nodes}
print('clusters_inferred 2nd (dependent) iteration', node2cluster_inferred)
