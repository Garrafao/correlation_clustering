import networkx as nx
from itertools import combinations
from correlation import cluster_correlation_search
from utils import get_clusters, transform_edge_weights
import numpy as np
from correlation import Loss
import pickle

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
clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_iter = 200)

# Display results
node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
node2cluster_inferred = {node:node2cluster_inferred[node] for node in nodes}
print('clusters_inferred', node2cluster_inferred)
print('loss', cluster_stats['loss'])

# Clustering again and initializing with the previous solution can improve the solution in many cases (this can be done multiple times)
clusters, cluster_stats = cluster_correlation_search(graph, s = 5, max_iter = 50, initial = clusters)

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

# Test loss replication on public data

## DWUG DE 3.0.0, large sparse graphs with quasi-ordinal edge weights
threshold = 2.5
for filename, loss_public in [('Abgesang', 108.5), ('Kubikmeter', 0), ('Titel', 147)]: # Titel is hard, its public loss is 138, but when keeping nan edges and noise cluster, it is 147
    runtime = 0
    with open('data/' + filename, 'rb') as f:
        graph = pickle.load(f)
    weight_transformation = lambda x: x-threshold
    graph = transform_edge_weights(graph, transformation = weight_transformation) # shift edge weights

    # Cluster graph
    clusters = []
    for i in range(5):
        clusters, cluster_stats = cluster_correlation_search(graph, s = 20, max_iter = 50, initial = clusters)
        loss = cluster_stats['loss']
        runtime += cluster_stats['runtime']
        print('  Intermediate loss:', loss)
        if loss <= loss_public:
            break
    print('loss', loss, 'loss_public', loss_public, 'runtime', runtime)
    assert loss <= loss_public

## SweWUG 2.0.0, small dense graphs with dense edge weights
threshold = 0.6
for filename, loss_public in [('al', 1.43347366), ('privatsak', 0), ('styvbarn', 2.77383317)]:
    runtime = 0
    with open('data/' + filename, 'rb') as f:
        graph = pickle.load(f)
    weight_transformation = lambda x: x-threshold
    graph = transform_edge_weights(graph, transformation = weight_transformation) # shift edge weights

    # Cluster graph
    clusters = []
    for i in range(5):
        clusters, cluster_stats = cluster_correlation_search(graph, s = 20, max_iter = 50, initial = clusters)
        loss = cluster_stats['loss']
        runtime += cluster_stats['runtime']
        print('  Intermediate loss:', loss)
        if np.isclose(loss, loss_public):
            break
    print('loss', loss, 'loss_public', loss_public, 'runtime', runtime)
    assert np.isclose(loss, loss_public)



