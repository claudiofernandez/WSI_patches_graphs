import os
import openslide
import torch
import numpy as np
import matplotlib.pyplot as plt



# Load a specific region of a WSI using OpenSlide
def load_wsi_region(wsi_path, region_coords, region_size):
    """
    Load a region of the WSI efficiently based on coordinates.
    Args:
        wsi_path: Path to the WSI file.
        region_coords: The (x, y) coordinates of the region to load.
        region_size: Size of the region (width, height).
    """
    slide = openslide.OpenSlide(wsi_path)
    region = slide.read_region(region_coords, 0, region_size)  # 0 is the zoom level
    slide.close()
    return region


# Load the graph and patch coordinates
def load_graph(graph_path):
    """
    Load the graph data, including patch coordinates.
    This assumes the graph contains a 'centroid' field with patch coordinates.
    """
    graph_data = torch.load(graph_path)
    patch_coords = graph_data['centroid']  # Assuming centroids are stored as (x, y)
    return graph_data, patch_coords


# Visualize attention on the WSI
def visualize_attention_on_wsi(wsi_path, graph_path, model, patch_size=224, downscale_factor=1):
    """
    Visualize attention scores on WSI by overlaying heatmap on patch regions.

    Args:
        wsi_path: Path to the WSI file.
        graph_path: Path to the graph file.
        model: Trained PatchGCN model.
        patch_size: Size of each patch.
        downscale_factor: Factor to downscale the WSI for visualization.
    """
    # Load the WSI and graph
    graph, patch_coords = load_graph(graph_path)

    # Adjust patch coordinates if downscaling
    if downscale_factor != 1:
        patch_coords = [(int(x / downscale_factor), int(y / downscale_factor)) for (x, y) in patch_coords]

    # Load the WSI and extract regions
    wsi_image = openslide.OpenSlide(wsi_path)

    # Perform inference with the model to get attention scores
    model.eval()
    with torch.no_grad():
        _, _, _, h = model(graph, pool='attention')  # Forward pass with attention pooling
        A_path, _ = model.path_attention_head(h)
        attention_scores = torch.transpose(A_path, 1, 0).squeeze().cpu().numpy()

    # Normalize attention scores
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())

    # Create a blank canvas to overlay attention heatmap
    heatmap = np.zeros((int(wsi_image.dimensions[1] / downscale_factor),
                        int(wsi_image.dimensions[0] / downscale_factor)))

    # Loop through each patch and overlay attention scores
    for i, (x, y) in enumerate(patch_coords):
        region = load_wsi_region(wsi_path, (x, y), (patch_size, patch_size))
        heatmap[y:y + patch_size, x:x + patch_size] = attention_scores[i]

    # Display the heatmap overlayed on the downscaled WSI
    downscaled_wsi = wsi_image.get_thumbnail((wsi_image.dimensions[0] // downscale_factor,
                                              wsi_image.dimensions[1] // downscale_factor))
    downscaled_wsi = np.array(downscaled_wsi.convert("RGB"))

    plt.figure(figsize=(10, 10))
    plt.imshow(downscaled_wsi)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
    plt.colorbar(label='Attention Score')
    plt.title("Attention Heatmap on WSI")
    plt.show()
    print("hola")


# Example usage
wsi_path = "E:/CLARIFY BREAST CANCER DATABASE/WSIS CLARIFY FULL BCDB RECONSTRUCTED/SUS001-2021-09-24_13.39.24.tif"
graph_path = "C:/Users/clferma1/Documents/Investigacion_GIT/WSI_patches_graphs/data/BCNB/results_graphs_november_23/graphs_PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_MAGN_10x/graphs_k_25/1_graph.pt"
model = torch.load("C:/Users/clferma1/Documents/Investigacion_GIT/WSI_patches_graphs/data/gcn_pretrained_models/[24_11_2023]_GCN_Final_BCNB_OTHERvsTNBC_GT_GENConv_GL_5_KNN_19_EA_spatial_EF_False_GP_mean_DO_True_LR_1e-05.pth")  # Load your trained PatchGCN model
visualize_attention_on_wsi(wsi_path, graph_path, model, patch_size=224, downscale_factor=16)

print("hola")
