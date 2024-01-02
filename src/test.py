import networkx as nx
from itertools import combinations
from correlation import cluster_correlation_search
from utils import get_clusters, transform_edge_weights
import numpy as np
from correlation import Loss 

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
print('loss', cluster_stats['loss'])

# Clustering again and initializing with the previous solution can improve the solution in many cases (this can be done multiple times)
clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_attempts = 100, max_iters = 200, initial = clusters)

# Display results after second iteration
node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
node2cluster_inferred = {node:node2cluster_inferred[node] for node in nodes}
print('clusters_inferred 2nd (dependent) iteration', node2cluster_inferred)
print('loss', cluster_stats['loss'])

# Example how to get linear loss for predefined cluster solution    
n2i = {node:i for i, node in enumerate(graph.nodes())}
n2c = {n2i[node]:i for i, cluster in enumerate(clusters) for node in cluster}
edges_positive = set([(n2i[i],n2i[j],w-0.0) for (i,j,w) in graph.edges.data("weight") if w >= 0.0])
edges_negative = set([(n2i[i],n2i[j],w-0.0) for (i,j,w) in graph.edges.data("weight") if w < 0.0])    
cluster_state = np.array([n2c[n] for n in sorted(n2c.keys())])  
loss = Loss('linear_loss', edges_positive=edges_positive, edges_negative=edges_negative).loss(cluster_state)
assert loss == cluster_stats['loss']
