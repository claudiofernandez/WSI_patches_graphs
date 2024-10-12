import torch
import numpy as np
import matplotlib.pyplot as plt


# Mockup WSI image and patch coordinates
def create_mock_wsi_image(image_size=(500, 500), num_patches=50):
    """
    Create a mock WSI image with random patches and coordinates.
    """
    # Mock WSI: A blank grayscale image
    wsi_image = np.ones(image_size) * 255  # White background (grayscale image)

    # Random patch coordinates (mocking the patch coordinates in a smaller resolution WSI)
    np.random.seed(42)
    patch_coords = np.random.randint(0, min(image_size) - 50, size=(num_patches, 2))

    return wsi_image, patch_coords


# Mockup model to generate attention scores
def mock_attention_scores(num_patches):
    """
    Create random attention scores for the mockup.
    """
    attention_scores = np.random.rand(num_patches)
    attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
    return attention_scores


# Visualization function for overlaying attention scores on WSI image
def visualize_attention_mockup(wsi_image, patch_coords, attention_scores, patch_size=50):
    """
    Visualize the attention scores overlayed on a mock WSI image.

    Args:
        wsi_image: Mock WSI image (2D grayscale).
        patch_coords: List of (x, y) coordinates of patches.
        attention_scores: Normalized attention scores for each patch.
        patch_size: The size of the patches for visualization.
    """
    # Create an empty heatmap of the same size as the WSI
    heatmap = np.zeros_like(wsi_image)

    # Overlay the attention scores onto the corresponding patches
    for i, (x, y) in enumerate(patch_coords):
        # Draw a patch with attention score
        heatmap[y:y + patch_size, x:x + patch_size] = attention_scores[i]

    # Plot the original WSI with the attention heatmap overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(wsi_image, cmap='gray', interpolation='nearest')
    plt.imshow(heatmap, cmap='jet', alpha=0.5, interpolation='nearest')  # Overlay heatmap with transparency
    plt.colorbar(label='Attention Score')
    plt.title("Mockup Attention Heatmap over WSI")
    plt.axis('off')
    plt.show()


# Mockup execution
if __name__ == "__main__":
    # Create a mock WSI image and random patch coordinates
    wsi_image, patch_coords = create_mock_wsi_image(image_size=(300, 300), num_patches=20)

    # Generate mock attention scores
    attention_scores = mock_attention_scores(num_patches=len(patch_coords))

    # Visualize the attention scores overlayed on the mock WSI image
    visualize_attention_mockup(wsi_image, patch_coords, attention_scores, patch_size=40)
