import time
import os
from tqdm import tqdm
import argparse
import torch
import ast
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from graphs_utils import *


class embeddings2graph_w_edgefeatures_norm(torch.nn.Module):
    def __init__(self, knn=8, **kwargs):
        """
        Simplified graph creator that works directly with pre-computed embeddings.
        Args:
            knn (int): Number of nearest neighbors for graph construction
            **kwargs: Additional arguments (kept for compatibility but not used)
        """
        super(embeddings2graph_w_edgefeatures_norm, self).__init__()
        self.knn = knn

    def forward(self, embeddings, img_coords, batch_size=1):
        """
        Create a graph from pre-computed embeddings and coordinates.
        Args:
            embeddings (torch.Tensor): Pre-computed embeddings
            img_coords (np.ndarray): Coordinates of the patches
            batch_size (int): Kept for compatibility but not used
        Returns:
            geomData: Graph object containing node features, edge indices, and edge features
        """
        # Convert inputs to numpy if they're not already
        coords = np.array(img_coords)
        features = np.array(embeddings.cpu().detach() if isinstance(embeddings, torch.Tensor) else embeddings)

        # Verify shapes match
        assert coords.shape[0] == features.shape[0], "Number of coordinates must match number of features"
        num_patches = coords.shape[0]
        radius = self.knn + 1

        # Adjust radius if necessary
        if num_patches <= radius:
            radius = num_patches

        # Compute spatial distance (based on coordinates)
        model = Hnsw(space='l2')
        model.fit(coords)
        a = np.repeat(range(num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:]
                                for v_idx in range(num_patches)]), dtype=int)
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Compute latent distance (based on embeddings)
        model = Hnsw(space='l2')
        model.fit(features)
        a = np.repeat(range(num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(features[v_idx], topn=radius)[1:]
                                for v_idx in range(num_patches)]), dtype=int)
        edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Compute edge_features (normalized euclidean distance between coordinates)
        max_coord = np.array([np.max(coords, axis=0)])
        min_coord = np.array([np.min(coords, axis=0)])
        norm_coords = (coords - min_coord) / (max_coord - min_coord)

        edge_features = torch.zeros((edge_spatial.shape[1],))
        for i, (idx1, idx2) in enumerate(edge_spatial.t().tolist()):
            coord1, coord2 = norm_coords[idx1], norm_coords[idx2]
            euclidean_distance = np.sqrt(np.sum((coord1 - coord2) ** 2))
            edge_features[i] = euclidean_distance

        # Create and return the graph
        G = geomData(
            x=torch.Tensor(features),
            edge_index=edge_spatial,
            edge_latent=edge_latent,
            edge_features=edge_features,
            centroid=torch.Tensor(coords)
        )

        return G


def analyze_graph(graph_path):
    """Analyze a single graph and display its statistics and visualizations."""
    # Load graph
    graph = torch.load(graph_path)

    # Extract basic statistics
    num_nodes = graph['x'].shape[0]
    num_edges = graph['edge_index'].shape[1]
    edge_features = graph['edge_features'].numpy()
    coords = graph['centroid'].numpy()

    # Print statistics
    print(f"\nGraph Statistics for {os.path.basename(graph_path)}:")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Mean edge distance: {np.mean(edge_features):.4f}")
    print(f"Median edge distance: {np.median(edge_features):.4f}")

    # Detect overlapping patches
    distances = squareform(pdist(coords))
    overlapping_pairs = np.sum(distances < 10) - len(coords)  # threshold of 10
    print(f"Number of overlapping patch pairs: {overlapping_pairs // 2}")

    # Create visualizations
    plt.figure(figsize=(15, 5))

    # Plot 1: Node distribution
    plt.subplot(1, 3, 1)
    plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.6)
    plt.title("Node Distribution")
    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")

    # Plot 2: Node feature distribution
    plt.subplot(1, 3, 2)
    node_features = graph['x'].numpy()
    plt.hist(node_features.flatten(), bins=50, alpha=0.7)
    plt.title("Node Feature Distribution")
    plt.xlabel("Feature value")
    plt.ylabel("Frequency")

    # Plot 3: Edge distance distribution
    plt.subplot(1, 3, 3)
    plt.hist(edge_features, bins=50, alpha=0.7)
    plt.title("Edge Distance Distribution")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'mean_edge_dist': np.mean(edge_features),
        'median_edge_dist': np.median(edge_features),
        'overlapping_pairs': overlapping_pairs // 2
    }


def compare_graphs(graph_paths):
    """Compare multiple graphs and their statistics."""
    if not isinstance(graph_paths, list):
        graph_paths = [graph_paths]

    stats = []
    for path in graph_paths:
        stat = analyze_graph(path)
        stats.append(stat)

    # If comparing multiple graphs, show comparative statistics
    if len(stats) > 1:
        print("\nComparative Statistics:")
        metrics = ['num_nodes', 'num_edges', 'mean_edge_dist', 'median_edge_dist', 'overlapping_pairs']

        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(metrics):
            values = [stat[metric] for stat in stats]
            plt.subplot(1, 5, i + 1)
            plt.bar(range(len(values)), values)
            plt.title(metric)
            plt.xticks(range(len(values)), [f'Graph {i + 1}' for i in range(len(values))])
        plt.tight_layout()
        plt.show()


def read_assets_from_npy(embeddings_path, coords_path):
    """Read embeddings and coordinates from .npy files"""
    embeddings = np.load(embeddings_path)
    coords = np.load(coords_path)
    return embeddings, coords


def create_graph(knn_list, embeddings, coords, wsi_name, dir_results_save_graph):
    """Create graph from embeddings and coordinates"""
    # Convert embeddings to tensor if needed
    embeddings_tensor = torch.tensor(embeddings) if not isinstance(embeddings, torch.Tensor) else embeddings

    # Round coordinates to match previous patching method
    coords = np.round(coords / 512).astype(int)

    # Create graph filename
    graph_savename = wsi_name + "_graph.pt"

    # Iterate over K list
    for k in knn_list:
        dir_folder_savegraphs_k = os.path.join(dir_results_save_graph, f"graphs_k_{k}")
        os.makedirs(dir_folder_savegraphs_k, exist_ok=True)

        save_path = os.path.join(dir_folder_savegraphs_k, graph_savename)
        if not os.path.isfile(save_path):
            # Create graph creator with just knn parameter
            graph_creator = imgs2graph_w_edgefeatures_norm(knn=k)

            # Generate graph from embeddings
            graph = graph_creator(embeddings=embeddings_tensor, img_coords=coords)

            # Save graph
            torch.save(graph, save_path)
            print(f"Graph saved for {wsi_name} with k={k}")

            # Analyze graph if requested
            if args.analyze_graphs:
                print(f"\nAnalyzing graph for {wsi_name} with k={k}")
                try:
                    analyze_graph(save_path)
                except Exception as e:
                    print(f"Error analyzing graph: {str(e)}")


def main(args):
    # If analysis is enabled, make sure matplotlib works in the current environment
    if args.analyze_graphs:
        import matplotlib
        if os.environ.get('DISPLAY', '') == '':
            print('No display found. Using non-interactive Agg backend')
            matplotlib.use('Agg')
    # Use parent directory structure
    parent_dir = args.parent_dir

    # Define subdirectories
    dir_embeddings = os.path.join(parent_dir, "embeddings")
    dir_coords = os.path.join(parent_dir, "coords")
    dir_results_save_graph = os.path.join(parent_dir, "output_graphs")

    # Verify embeddings and coords directories exist
    if not os.path.exists(dir_embeddings) or not os.path.exists(dir_coords):
        raise ValueError(f"Required directories 'embeddings' and 'coords' must exist in {parent_dir}")

    # Create output directory
    os.makedirs(dir_results_save_graph, exist_ok=True)

    print("Working directories:")
    print(f"Embeddings: {dir_embeddings}")
    print(f"Coordinates: {dir_coords}")
    print(f"Output graphs: {dir_results_save_graph}")

    # Get list of files
    embedding_files = sorted([f for f in os.listdir(dir_embeddings) if f.endswith('.npy')])

    # Process each file
    for embedding_file in tqdm(embedding_files):
        # Get corresponding coords file
        coords_file = embedding_file  # Same filename but in coords directory

        # Get WSI name from filename
        wsi_name = embedding_file.split('.')[0]  # Remove .npy extension

        # Derive full file paths
        embedding_path = os.path.join(dir_embeddings, embedding_file)
        coords_path = os.path.join(dir_coords, coords_file)

        print(f"\nProcessing {wsi_name}...")
        start_time = time.time()

        # Read files
        try:
            embeddings, coords = read_assets_from_npy(embedding_path, coords_path)
        except Exception as e:
            print(f"Error reading files for {wsi_name}: {str(e)}")
            continue

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Files read in {elapsed_time:.2f} seconds")

        # Generate graphs
        create_graph(
            knn_list=args.knn_list,
            embeddings=embeddings,
            coords=coords,
            wsi_name=wsi_name,
            dir_results_save_graph=dir_results_save_graph
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Directory structure
    parser.add_argument('--parent_dir', type=str, default="C:/Users/clferma1/Documents/Investigacion_GIT/CPathPipeline2025/CPathPipeline/output/graphs_16_01_2025_dws_4_conchi",
                        help='Parent directory containing embeddings/ and coords/ folders')

    # Graph parameters
    parser.add_argument("--knn_list", default=[8, 19, 25], type=lambda x: eval(x) if isinstance(x, str) else x,
                        help='KNN values for generating graphs (e.g. [8,19,25])')
    # Analysis parameters
    parser.add_argument("--analyze_graphs", action="store_true",
                        help="Enable graph analysis and visualization")
    parser.add_argument("--save_plots", action="store_true",
                        help="Save analysis plots instead of displaying them")
    parser.add_argument("--plots_dir", type=str, default=None,
                        help="Directory to save analysis plots (default: parent_dir/analysis_plots)")

    args = parser.parse_args()
    main(args)