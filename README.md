# Correlation Clustering

A python implementation of Correlation Clustering [(Bansal et al., 2004)](https://link.springer.com/article/10.1023/B:MACH.0000033116.57574.95). Correlation Clustering is a weighted graph clustering technique minimizing the sum of cluster disagreements, i.e., the sum of negative edge weights within clusters plus the sum of positive edge weights across clusters. It has some nice properties, e.g.:

- finds number of clusters by itself
- handles missing edges
- robust to errors by minimizing a global loss
- optimizes an intuitive quality criterion
- our implementation is fast by using multiprocessing

If you use this software for academic research, please [cite](#bibtex) these papers:

- Dominik Schlechtweg, Nina Tahmasebi, Simon Hengchen, Haim Dubossarsky, Barbara McGillivray. 2021. [DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages](https://aclanthology.org/2021.emnlp-main.567/). In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing.
- Dominik Schlechtweg. 2023. [Human and Computational Measurement of Lexical Semantic Change](http://dx.doi.org/10.18419/opus-12833). PhD thesis. University of Stuttgart.

Find further extensive experiments testing and optimizing v1.0.0 of this implementation in:

- Benjamin Tunc. [Optimierung von Clustering von Wortverwendungsgraphen](https://elib.uni-stuttgart.de/handle/11682/11923). Bachelor thesis. University of Stuttgart. [[slides]](https://garrafao.github.io/publications/211201-optimierung-wugs.pdf)

### Simple example
```
import networkx as nx
from itertools import combinations
from correlation_clustering.correlation import cluster_correlation_search
import numpy as np

# Define true clusters
nodes = ['node1', 'node2', 'node3', 'node4']
node2clusters_true = {'node1':0, 'node2':0, 'node3':1, 'node4':1}
print('clusters_true', node2clusters_true)

# Initialize graph
graph = nx.Graph()

# Generate perfectly clusterable graph
for (u,v) in combinations(nodes, 2):
    if node2clusters_true[u] == node2clusters_true[v]:
        graph.add_edge(u, v, weight=np.random.choice([3,4]))
    else:
        graph.add_edge(u, v, weight=np.random.choice([1,2]))

# Prepare graph for clustering
threshold = 2.5
for (i,j) in graph.edges():
    graph[i][j]['weight'] = graph[i][j]['weight']-threshold # shift edge weights

# Cluster graph
clusters, cluster_stats = cluster_correlation_search(graph)

# Display results
node2cluster_inferred = {node:i for i, cluster in enumerate(clusters) for node in cluster}
node2cluster_inferred = {node:node2cluster_inferred[node] for node in nodes}
print('clusters_inferred', node2cluster_inferred)
print('loss', cluster_stats['loss'])
```

### Installation

To install the package run

	pip install correlation-clustering

Please run the test script with

	pytest correlation_clustering/tests/


BibTex
--------

```
@inproceedings{Schlechtweg2021dwug,
 title = {{DWUG}: A large Resource of Diachronic Word Usage Graphs in Four Languages},
 author = {Schlechtweg, Dominik  and Tahmasebi, Nina  and Hengchen, Simon  and Dubossarsky, Haim  and McGillivray, Barbara},
 booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
 publisher = {Association for Computational Linguistics},
 address = {Online and Punta Cana, Dominican Republic},
 pages = {7079--7091},
 url = {https://aclanthology.org/2021.emnlp-main.567},
 year = {2021}
}
```
```
@phdthesis{Schlechtweg2023measurement,
  author  = {Schlechtweg, Dominik},
  title   = {Human and Computational Measurement of Lexical Semantic Change},
  school  = {University of Stuttgart},
  address =  {Stuttgart, Germany},
  year    = {2023},
  url = {http://dx.doi.org/10.18419/opus-12833},
  slides = {https://garrafao.github.io/publications/220324-thesis-slides.pdf}
}
```
```
@mastersthesis{Tunc2021OptimierungWUGs,
author = {Benjamin Tunc},
year = {2021}, 
title = {{Optimierung von Clustering von Wortverwendungsgraphen}},
type = {Bachelor thesis},
school = {University of Stuttgart},
slides = {https://garrafao.github.io/publications/211201-optimierung-wugs.pdf},
url = {https://elib.uni-stuttgart.de/handle/11682/11923}
}
```



