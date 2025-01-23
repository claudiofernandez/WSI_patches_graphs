import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity



def load_graph(graph_path):
    """Load a graph from a .pt file."""
    graph = torch.load(graph_path)
    return graph


def get_graph_statistics(graph):
    """Extract and return basic statistics of the graph."""
    num_nodes = graph['x'].shape[0]  # Number of nodes
    num_edges = graph['edge_index'].shape[1]  # Number of edges
    edge_distances = graph['edge_features'].numpy()  # Edge features (distances)

    return num_nodes, num_edges, edge_distances


def detect_overlapping_patches(coords, threshold=10):
    """Detect overlapping patches based on a distance threshold.
    Args:
        coords (numpy array): Array of patch coordinates.
        threshold (int): Distance threshold to consider patches overlapping.
    Returns:
        int: Number of overlapping pairs.
    """
    distances = squareform(pdist(coords))  # Compute pairwise distances
    overlapping_pairs = np.sum(distances < threshold) - len(coords)  # Exclude self-loops
    return overlapping_pairs // 2  # Each pair is counted twice


def compare_features(graph_bcn, graph_clarify):
    """Compare node and edge features between BCNB and CLARIFY graphs."""
    # Node feature comparison
    node_features_bcn = graph_bcn['x'].numpy()
    node_features_clarify = graph_clarify['x'].numpy()

    print("Node Features:")
    print(f"BCNB - Mean: {np.mean(node_features_bcn):.4f}, Variance: {np.var(node_features_bcn):.4f}")
    print(f"CLARIFY - Mean: {np.mean(node_features_clarify):.4f}, Variance: {np.var(node_features_clarify):.4f}\n")

    # Plot node feature distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(node_features_bcn.flatten(), bins=50, alpha=0.7, label='BCNB')
    plt.hist(node_features_clarify.flatten(), bins=50, alpha=0.7, label='CLARIFY')
    plt.legend()
    plt.title("Node Feature Distribution")

    # Edge feature comparison
    edge_features_bcn = graph_bcn['edge_features'].numpy()
    edge_features_clarify = graph_clarify['edge_features'].numpy()

    print("Edge Features:")
    print(f"BCNB - Mean: {np.mean(edge_features_bcn):.4f}, Variance: {np.var(edge_features_bcn):.4f}")
    print(f"CLARIFY - Mean: {np.mean(edge_features_clarify):.4f}, Variance: {np.var(edge_features_clarify):.4f}\n")

    # Plot edge feature distributions
    plt.subplot(1, 2, 2)
    plt.hist(edge_features_bcn, bins=50, alpha=0.7, label='BCNB')
    plt.hist(edge_features_clarify, bins=50, alpha=0.7, label='CLARIFY')
    plt.legend()
    plt.title("Edge Feature Distribution")
    plt.show()

    # Cosine similarity comparison
    cosine_sim = cosine_similarity(node_features_bcn, node_features_clarify)
    plt.figure(figsize=(8, 6))
    plt.imshow(cosine_sim, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Cosine Similarity Between Node Features")
    plt.show()


def compare_graphs(graph_path_bcn, graph_path_clarify):
    """Compare graphs from BCNB and CLARIFY datasets."""

    # Load graphs
    graph_bcn = load_graph(graph_path_bcn)
    graph_clarify = load_graph(graph_path_clarify)

    # Get statistics for both graphs
    stats_bcn = get_graph_statistics(graph_bcn)
    stats_clarify = get_graph_statistics(graph_clarify)

    print("BCNB Graph Statistics:")
    print(f"Number of nodes: {stats_bcn[0]}")
    print(f"Number of edges: {stats_bcn[1]}")
    print(f"Mean edge distance: {np.mean(stats_bcn[2]):.4f}")
    print(f"Median edge distance: {np.median(stats_bcn[2]):.4f}\n")

    print("CLARIFY Graph Statistics:")
    print(f"Number of nodes: {stats_clarify[0]}")
    print(f"Number of edges: {stats_clarify[1]}")
    print(f"Mean edge distance: {np.mean(stats_clarify[2]):.4f}")
    print(f"Median edge distance: {np.median(stats_clarify[2]):.4f}\n")

    # Detect overlapping patches
    coords_bcn = graph_bcn['centroid'].numpy()
    coords_clarify = graph_clarify['centroid'].numpy()

    overlapping_bcn = detect_overlapping_patches(coords_bcn)
    overlapping_clarify = detect_overlapping_patches(coords_clarify)

    print(f"Number of overlapping patch pairs in BCNB: {overlapping_bcn}")
    print(f"Number of overlapping patch pairs in CLARIFY: {overlapping_clarify}")

    # Plot node distributions for visual comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(coords_bcn[:, 0], coords_bcn[:, 1], s=10)
    plt.title("BCNB Node Distribution")

    plt.subplot(1, 2, 2)
    plt.scatter(coords_clarify[:, 0], coords_clarify[:, 1], s=10)
    plt.title("CLARIFY Node Distribution")
    plt.show()

    # Compare features
    compare_features(graph_bcn, graph_clarify)


# Example usage
if __name__ == "__main__":
    bcn_graph_path = "../data/BCNB/results_graphs_november_23/graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x/graphs_k_8/4_graph.pt" #1_graph.pt
    clarify_graph_path = "../data/CLARIFY/results_graphs_november_23/graphs_PM_LUMINALAvsLUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OP/graphs_k_8/SUS004-2022-06-1316.27.47_graph.pt" #SUS001-2021-09-24_13.39.24_graph.pt
    compare_graphs(bcn_graph_path, clarify_graph_path)
