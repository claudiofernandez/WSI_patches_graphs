import h5py
import time
import os
from tqdm import tqdm
import argparse
import torch
import pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from graphs_utils import *

def main(args):

    # # Set up directories depending on where this program is executed
    # if args.where_exec == "slurm_nas":
    #     dir_results_save_graph = os.path.join('/workspace/NASFolder', args.graph_savedir)
    #     dir_h5s = os.path.join('/workspace/NASFolder', args.path_to_h5files)
    #     feature_extractor_dir = os.path.join('/workspace/NASFolder', args.path_to_feature_extractors_folder)
    # elif args.where_exec == "slurm_dgx":
    #     dir_results_save_graph = os.path.join('/workspace/NASFolder', args.graph_savedir)
    #     dir_h5s = os.path.join('/workspace/DGXFolder', args.path_to_h5files)
    #     feature_extractor_dir = os.path.join('/workspace/DGXFolder', args.path_to_feature_extractors_folder)
    # elif args.where_exec == "dgx_gpu":
    #     dir_results_save_graph = os.path.join('/workspace/exec/NASFolder', args.graph_savedir)
    #     dir_h5s = os.path.join('/workspace/exec/dataDGX', args.path_to_h5files)
    #     feature_extractor_dir = os.path.join('/workspace/exec/NASFolder', args.path_to_feature_extractors_folder)
    # elif args.where_exec == "local":
    #     dir_results_save_graph = os.path.join('./', args.graph_savedir)
    #     dir_data = os.path.join('C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB', args.path_to_h5files)
    #     feature_extractor_dir = os.path.join('C:/Users/clferma1/Documents/Investigacion_GIT', args.path_to_feature_extractors_folder)

    os.makedirs(args.savedir, exist_ok=True) # create output directory

    # Read ground truth
    gt_df = pd.read_excel("./data/BCNB/ground_truth/patient-clinical-data.xlsx")

    # Read patches paths class perc
    dir_excels_class_perc = "./data/BCNB/patches_paths_class_perc"

    # Read the excel including the images paths and their tissue percentage
    train_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc, "train_patches_class_perc_0_tp.csv"))
    val_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc,  "val_patches_class_perc_0_tp.csv"))
    test_class_perc_patches_paths_df = pd.read_csv(os.path.join(dir_excels_class_perc, "test_patches_class_perc_0_tp.csv"))

    # Merge 3 dataframes in 1
    frames = [train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df]

    # Concatenate the dataframes vertically
    merged_df = pd.concat(frames, ignore_index=True)

    # Select only ppatches that contain more than 40% tissue
    tissue_percentages_max = "O_0.4-T_1-S_1-I_1-N_1"

    class_perc_0_max = float(tissue_percentages_max.split("-")[0].split("_")[-1])
    class_perc_1_max = float(tissue_percentages_max.split("-")[1].split("_")[-1])
    class_perc_2_max = float(tissue_percentages_max.split("-")[2].split("_")[-1])
    class_perc_3_max = float(tissue_percentages_max.split("-")[3].split("_")[-1])
    class_perc_4_max = float(tissue_percentages_max.split("-")[4].split("_")[-1])

    # Filter and extract patches paths based on tissue percentage
    filtered_rows = merged_df.query("class_perc_0 <= " + str(class_perc_0_max) +
                                                "and class_perc_1 <= " + str(class_perc_1_max) +
                                                "and class_perc_2 <= " + str(class_perc_2_max) +
                                                "and class_perc_3 <= " + str(class_perc_3_max) +
                                                "and class_perc_4 <= " + str(class_perc_4_max))

    selected_images_paths = list(filtered_rows["patch_path"])

    # Graph dir
    graphs_dir = "C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB/results_graphs_november_23"

    #TODO: Extract features from graphs

    for root, dirs, files in os.walk(graphs_dir):
        # Extract feature extractor name from the current directory
        graphs_subfolder = os.path.basename(root)
        for file_name in files:
            if file_name.endswith(".pt"):
                file_id = file_name.split("_")[0]
                feature_extractor = os.path.basename(os.path.dirname(root))
                file_path = os.path.join(root, file_name)

                # Now you can use feature_extractor, subfolder, and file_path as needed
                print("Feature Extractor:", feature_extractor)
                print("Subfolder:", graphs_subfolder)
                print("File:", file_name)
                print("File Path:", file_path)

                # Load graph_data
                graph = torch.load(file_path)
                graph_features = graph["x"]
                graph_coords = graph["centroid"]

                # Assert that the len of the patches with tissue for that ID corresponds with the number of nodes in the graphs
                filtered_paths = [path for path in selected_images_paths if file_id in path.split("/")[-2]]

                assert len(graph_features) == len(filtered_paths)

                #TODO: Find matching label for graph



                print("\n")



    # Load pretrained model
    feature_extractor_dir = os.path.join("C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/feature_extractors")
    feature_extractor_path = os.path.join(feature_extractor_dir , args.feature_extractor_name)

    model = torch.load(feature_extractor_path)

    # get all features from graphs


    # pool them using MILAggregation

    # read BCNB dataset

    bb.eval()

    # Assuming you have your feature vectors and corresponding labels
    # Replace these with your actual feature vectors and labels
    feature_vectors = your_feature_vectors
    labels = your_labels

    # Assuming you have three different classification tasks
    tasks = ["LUMINALAvsLAUMINALBvsHER2vsTNBC", "LUMINALSvsHER2vsTNBC", "OTHERvsTNBC"]

    # Create a dictionary to map labels to colors
    label_color_map = {
        "LUMINALAvsLAUMINALBvsHER2vsTNBC": {"LUMINAL A": "red", "LUMINAL B": "blue", "HER2": "green", "TNBC": "purple"},
        "LUMINALSvsHER2vsTNBC": {"LUMINAL S": "orange", "HER2": "cyan", "TNBC": "brown"},
        "OTHERvsTNBC": {"OTHER": "pink", "TNBC": "gray"}
    }

    # Iterate through each classification task
    for task in tasks:
        # Extract feature vectors and labels for the current task
        task_indices = [i for i, label in enumerate(labels) if label in label_color_map[task]]
        task_feature_vectors = feature_vectors[task_indices]
        task_labels = [labels[i] for i in task_indices]

        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(task_feature_vectors)

        # Create a DataFrame for plotting
        df_tsne = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
        df_tsne['Labels'] = task_labels

        # Plot t-SNE with seaborn
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Labels', palette=label_color_map[task], data=df_tsne)
        plt.title(f't-SNE Plot for {task}')
        plt.show()


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--path_to_bcnb_dataset', type=str, default="C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB", help="path where h5 files are located")
    parser.add_argument('--path_to_feature_extractors_folder', type=str, default="/Molecular_Subtype_Prediction/data/feature_extractors", help="path where feature extractors are located")
    parser.add_argument('--savedir', type=str, default="/output_tsnes", help="path to save graphs folder")

    # Feature extractor params
    parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str, help='Classification task')
    parser.add_argument("--feature_extractor_name", default="PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10x_N_100_Anetwork_weights_best_f1.pth", type=str, help='Chosen feature extractor.')

    args = parser.parse_args()
    main(args)

print("hola")