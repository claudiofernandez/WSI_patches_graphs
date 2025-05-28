import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import torch
import os
import re
from collections import defaultdict


def find_models_flexible(directory, task_name, model_type="model"):
    """
    Flexible model finding that handles spelling variations
    """
    if not os.path.exists(directory):
        raise ValueError(f"Directory does not exist: {directory}")

    files = os.listdir(directory)
    print(f"Looking in {directory}")
    print(f"Available files: {files[:5]}...")  # Show first 5 files

    # Try exact match first
    exact_matches = [f for f in files if task_name in f]
    if exact_matches:
        print(f"Found {model_type} with exact match: {exact_matches[0]}")
        return exact_matches

    # Try common spelling variations
    variations = [
        task_name.replace("LUMINAL", "LAUMINAL"),
        task_name.replace("LAUMINAL", "LUMINAL"),
        task_name.replace("vs", "VS"),
        task_name.replace("VS", "vs"),
    ]

    for variation in variations:
        matches = [f for f in files if variation in f]
        if matches:
            print(f"Found {model_type} with variation '{variation}': {matches[0]}")
            return matches

    # If still no matches, show what's available for debugging
    print(f"No matches found for task: {task_name}")
    print("Available files contain these patterns:")
    unique_patterns = set()
    for f in files:
        if "LUMINAL" in f.upper() or "TNBC" in f.upper():
            # Extract the task-like part
            parts = f.split("_")
            for part in parts:
                if "LUMINAL" in part.upper() or "TNBC" in part.upper():
                    unique_patterns.add(part)
    print(f"Patterns found: {list(unique_patterns)}")

    return []


def plot_nca_ca_tsne_comparison(
        gt_path="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx",
        graphs_dir="../data/CLARIFY/results_graphs_january_25",
        mil_models_dir="../data/feature_extractors",
        gcn_models_dir="../data/gcn_pretrained_models",
        output_dir="./tsne_plots",
        task_name="LUMINALAvsLUMINALBvsHER2vsTNBC",
        knn=25,
        perplexity=30,
        random_state=42
):
    """
    Create t-SNE plots comparing NCA (MIL) and CA (GCN) feature representations
    for WSI-level features extracted from graph data.

    Args:
        gt_path: Path to ground truth Excel file
        graphs_dir: Directory containing graph data
        mil_models_dir: Directory containing pretrained MIL models
        gcn_models_dir: Directory containing pretrained GCN models
        output_dir: Directory to save plots
        task_name: Classification task name
        knn: KNN value for graph construction
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load ground truth
    gt_df = pd.read_excel(gt_path)

    # Task label mappings
    tasks_labels_mappings = {
        "LUMINALAvsLUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
        "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
        "OTHERvsTNBC": {"Other": 0, "TNBC": 1}
    }

    # Color mappings for visualization
    color_mappings = {
        "LUMINALAvsLUMINALBvsHER2vsTNBC": {0: '#1f77b4', 1: '#ff7f0e', 2: '#2ca02c', 3: '#d62728'},
        "LUMINALSvsHER2vsTNBC": {0: '#1f77b4', 1: '#2ca02c', 2: '#d62728'},
        "OTHERvsTNBC": {0: '#1f77b4', 1: '#d62728'}
    }

    # Label names for legend
    label_names = {
        "LUMINALAvsLUMINALBvsHER2vsTNBC": {0: 'Luminal A', 1: 'Luminal B', 2: 'HER2(+)', 3: 'TNBC'},
        "LUMINALSvsHER2vsTNBC": {0: 'Luminal', 1: 'HER2(+)', 2: 'TNBC'},
        "OTHERvsTNBC": {0: 'Other', 1: 'TNBC'}
    }

    task_labels_mapping = tasks_labels_mappings[task_name]
    colors = color_mappings[task_name]
    labels = label_names[task_name]

    print(f"Processing task: {task_name}")

    # Handle different spelling patterns in MIL vs GCN models
    mil_task_name = task_name
    gcn_task_name = task_name

    if task_name == "LUMINALAvsLUMINALBvsHER2vsTNBC":
        mil_task_name = "LUMINALAvsLAUMINALBvsHER2vsTNBC"  # MIL models use LAUMINAL
        gcn_task_name = "LUMINALAvsLUMINALBvsHER2vsTNBC"  # GCN models use LUMINAL

    # Find models using specific corrected names
    mil_models = find_models_flexible(mil_models_dir, mil_task_name, "MIL")
    gcn_models = find_models_flexible(gcn_models_dir, gcn_task_name, "GCN")

    if not mil_models:
        raise ValueError(f"Could not find MIL models for task {mil_task_name}")
    if not gcn_models:
        raise ValueError(f"Could not find GCN models for task {gcn_task_name}")

    mil_model_path = os.path.join(mil_models_dir, mil_models[0])
    gcn_model_path = os.path.join(gcn_models_dir, gcn_models[0])

    print(f"MIL model: {mil_models[0]}")
    print(f"GCN model: {gcn_models[0]}")

    # Load models
    mil_model = torch.load(mil_model_path).to('cuda')
    gcn_model = torch.load(gcn_model_path).to('cuda')

    mil_model.eval()
    gcn_model.eval()

    # Find graphs directory using GCN task name (since graphs likely match GCN naming)
    graph_dirs = [d for d in os.listdir(graphs_dir) if gcn_task_name in d]
    if not graph_dirs:
        # Try MIL task name if GCN name doesn't work
        graph_dirs = [d for d in os.listdir(graphs_dir) if mil_task_name in d]
    if not graph_dirs:
        raise ValueError(f"Could not find graph directory for tasks {gcn_task_name} or {mil_task_name}")

    graphs_knn_dir = os.path.join(graphs_dir, graph_dirs[0], f"graphs_k_{knn}")
    graph_files = os.listdir(graphs_knn_dir)

    print(f"Using graphs directory: {graph_dirs[0]}")
    print(f"Found {len(graph_files)} graph files")

    # Extract patient IDs
    def extract_patient_id(filename):
        match = re.search(r'(SUS\d+)', filename)
        return match.group(1) if match else None

    # Create DataFrame of graph files
    graph_files_df = pd.DataFrame({
        'filename': graph_files,
        'SUS_number': [extract_patient_id(f) for f in graph_files]
    })

    # Merge with ground truth and filter
    merged_df = pd.merge(graph_files_df, gt_df, on='SUS_number', how='inner')
    filtered_df = merged_df[merged_df['Molsub_surr_7clf'] != 'Excluded']

    print(f"Processing {len(filtered_df)} samples")

    # Extract features
    nca_features = []
    ca_features = []
    all_labels = []
    patient_ids = []

    with torch.no_grad():
        for _, row in filtered_df.iterrows():
            graph_name = row['filename']
            patient_id = row['SUS_number']

            # Load graph
            file_path = os.path.join(graphs_knn_dir, graph_name)
            graph = torch.load(file_path).to('cuda')
            graph_features = graph["x"].to('cuda')

            # Extract NCA features (MIL aggregation)
            nca_feature = mil_model.milAggregation(graph_features)
            nca_features.append(nca_feature.cpu().numpy())

            # Extract CA features (GCN forward pass)
            _, _, _, ca_feature = gcn_model(graph)
            ca_features.append(ca_feature.cpu().numpy())

            # Get label
            label_str = row['Molsub_surr_4clf']
            encoded_label = task_labels_mapping.get(label_str, 0)
            all_labels.append(encoded_label)
            patient_ids.append(patient_id)

    # Convert to numpy arrays
    nca_features = np.vstack(nca_features)
    ca_features = np.vstack(ca_features)
    all_labels = np.array(all_labels)

    print(f"NCA features shape: {nca_features.shape}")
    print(f"CA features shape: {ca_features.shape}")
    print(f"Labels shape: {all_labels.shape}")

    # Compute t-SNE
    print("Computing t-SNE for NCA features...")
    tsne_nca = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    nca_2d = tsne_nca.fit_transform(nca_features)

    print("Computing t-SNE for CA features...")
    tsne_ca = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    ca_2d = tsne_ca.fit_transform(ca_features)

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot NCA (MIL) features
    for label_idx, label_name in labels.items():
        mask = all_labels == label_idx
        if np.any(mask):
            ax1.scatter(nca_2d[mask, 0], nca_2d[mask, 1],
                        c=colors[label_idx], label=f'{label_name} (n={np.sum(mask)})',
                        alpha=0.7, s=50)

    ax1.set_title(f'NCA (MIL Aggregation) - {task_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1', fontsize=12)
    ax1.set_ylabel('t-SNE Component 2', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Plot CA (GCN) features
    for label_idx, label_name in labels.items():
        mask = all_labels == label_idx
        if np.any(mask):
            ax2.scatter(ca_2d[mask, 0], ca_2d[mask, 1],
                        c=colors[label_idx], label=f'{label_name} (n={np.sum(mask)})',
                        alpha=0.7, s=50)

    ax2.set_title(f'CA (GCN + Attention) - {task_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1', fontsize=12)
    ax2.set_ylabel('t-SNE Component 2', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, f'tsne_comparison_{task_name}_knn{knn}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')

    print(f"Plot saved to: {output_path}")
    plt.show()

    # Create DataFrame with results for further analysis
    results_df = pd.DataFrame({
        'patient_id': patient_ids,
        'true_label': all_labels,
        'label_name': [labels[l] for l in all_labels],
        'nca_tsne_x': nca_2d[:, 0],
        'nca_tsne_y': nca_2d[:, 1],
        'ca_tsne_x': ca_2d[:, 0],
        'ca_tsne_y': ca_2d[:, 1]
    })

    # Save results
    results_path = os.path.join(output_dir, f'tsne_results_{task_name}_knn{knn}.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    # Print some statistics
    print("\n--- Feature Analysis ---")
    print(f"NCA feature statistics: mean={nca_features.mean():.4f}, std={nca_features.std():.4f}")
    print(f"CA feature statistics: mean={ca_features.mean():.4f}, std={ca_features.std():.4f}")

    # Calculate silhouette scores (requires scikit-learn)
    try:
        from sklearn.metrics import silhouette_score
        nca_silhouette = silhouette_score(nca_2d, all_labels)
        ca_silhouette = silhouette_score(ca_2d, all_labels)
        print(f"\nSilhouette Scores (higher = better clustering):")
        print(f"NCA (MIL): {nca_silhouette:.4f}")
        print(f"CA (GCN): {ca_silhouette:.4f}")
    except ImportError:
        print("Scikit-learn not available for silhouette score calculation")

    return results_df, nca_features, ca_features


# Example usage
if __name__ == "__main__":
    # Run for quaternary classification
    # results_df, nca_feat, ca_feat = plot_nca_ca_tsne_comparison(
    #     task_name="LUMINALAvsLUMINALBvsHER2vsTNBC",
    #     knn=25,
    #     perplexity=20
    # )

    # Optionally run for other tasks
    plot_nca_ca_tsne_comparison(task_name="LUMINALSvsHER2vsTNBC", knn=25, perplexity=20)
    plot_nca_ca_tsne_comparison(task_name="OTHERvsTNBC", knn=25, perplexity=20)

    print("hola")
