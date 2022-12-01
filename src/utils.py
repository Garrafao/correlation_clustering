import networkx as nx

def transform_edge_weights(G, transformation = lambda x: x):
    """
    Transform edge weights.       
    :param G: graph
    :param transformation: transformation function
    :return G: graph with transformed edges
    """

    G = G.copy()
    
    for (i,j) in G.edges():
        G[i][j]['weight'] = transformation(G[i][j]['weight'])
    
    return G

    
def get_clusters(G, is_include_noise = False, is_include_main = True, noise_label = -1):
    """
    Get clusters stored in graph.       
    :param G: graph
    :param is_include_noise: include noise cluster
    :param is_include_main: include main clusters
    :param noise_label: label for noise cluster
    :return clusters, c2n, n2c: clusters and mappings
    """

    c2n = defaultdict(lambda: [])
    n2c = {}
    for node in G.nodes():
        cluster = G.nodes()[node]['cluster']
        if cluster == noise_label and not is_include_noise:
            continue
        if cluster != noise_label and not is_include_main:
            continue
        c2n[cluster].append(node)
        n2c[node] = cluster

    c2n = dict(c2n)
    clusters = [set(c2n[c]) for c in c2n]
    clusters.sort(key=lambda x:-len(x)) # sort by size
        
    return clusters, c2n, n2c


def add_clusters(G, node2cluster, allow_missing=False, missing_label=-1):
    """
    Add clusters to graph.       
    :param G: graph
    :param allow_missing: whether to allow missing cluster labels for nodes
    :param node2cluster: mapping of nodes to clusters
    :return G: 
    """
    
    if set(G.nodes()) != set(node2cluster.keys()):
        if allow_missing:
            print('Warning: Cluster labels are missing for some nodes in graph.')
            node2cluster = node2cluster | {node:missing_label for node in G.nodes() if not node in node2cluster.keys()}
        else:
            sys.exit('Breaking: Cluster labels are missing for some nodes in graph.')
        
    nx.set_node_attributes(G, node2cluster, 'cluster')
        
    return G
