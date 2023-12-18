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
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import silhouette_score
import seaborn as sns
from graphs_utils import *
from tqdm import tqdm

def main(args):

    # os.makedirs(args.feats_savedir, exist_ok=True) # create output directory

    # Read ground truth
    #gt_df = pd.read_excel("../data/BCNB/ground_truth/patient-clinical-data.xlsx")
    gt_df = pd.read_excel("../data/CLARIFY/ground_truth/final_clinical_info_CLARIFY_DB.xlsx")

    # Read patches paths class perc
    dir_excels_class_perc = "../data/BCNB/patches_paths_class_perc"

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
    selected_images_paths = [path.replace("D:/", "F:/") for path in selected_images_paths]

    # RETRIEVE PATCHES PATHS FROM NAS
    dir_patches = "Z:/Shared_PFC-TFG-TFM/Claudio/MIL/MIL_receptors_local/data/patches_512_fullWSIs_0"


    # Graph dir


    # Define your classification tasks
    tasks_labels_mappings = {
        "LUMINALAvsLAUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "Triple negative": 3},
        "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "Triple negative": 2},
        "OTHERvsTNBC": {"Other": 0, "Triple negative": 1}
    }

    #Iterate over the graphs

    if args.context_aware == "NCA":
        args.feature_extractors_dir = "../data/feature_extractors"
        feature_extractors = os.listdir(args.feature_extractors_dir)
    elif args.context_aware == "CA":
        args.pretrained_gcn_models_dir = "../data/gcn_pretrained_models/"
        feature_extractors = os.listdir(args.pretrained_gcn_models_dir)
        print("hola")

    graphs_dirs = os.listdir(args.graphs_dir)

    for graph_dirname in graphs_dirs:

        # Depending on feature extractor select task name and labels
        fe_taskname = graph_dirname.split("_BB")[0].split("graphs_PM_")[1]
        task_labels_mapping = tasks_labels_mappings[fe_taskname]

        # Derive feature extractor used for extracting features to build the graphs
        # extracted_features_savepath = os.path.join(args.feats_savedir, )
        if args.context_aware == "NCA":
            chosen_model = [fe_name for fe_name in feature_extractors if fe_taskname in fe_name][0]
            chosen_model_path = os.path.join(args.feature_extractors_dir, chosen_model)

        elif args.context_aware == "CA":
            chosen_model = [fe_name for fe_name in feature_extractors if fe_taskname in fe_name][0]
            args.knn = chosen_model.split("KNN_")[1].split("_")[0]
            chosen_model_path = os.path.join(args.pretrained_gcn_models_dir, chosen_model)

        print("Chosen model: ", chosen_model)
        # Load pretrained model
        model = torch.load(chosen_model_path).to('cuda')

        #  Iterate over graphs
        graphs_knn_dir = os.path.join(args.graphs_dir, graph_dirname, "graphs_k_" + str(args.knn))
        print("Chosen graphs: ", graphs_knn_dir)
        graphs_files = os.listdir(graphs_knn_dir)

        # Initialize lists to store features, labels, and task labels
        all_features = []
        all_labels = []
        all_prediction_labels = []
        all_task_labels = []

        for graph_name in tqdm(graphs_files):

            #for task, labels_mappings in tasks_labels_mappings.items():

            max_n_cases = 5
            counter = 0

            file_id = graph_name.split("-")[0].split("HE")[0].split("_")[0].split("a")[0]
            feature_extractor = graph_dirname.split("graphs_PM_")[1]
            file_path = os.path.join(graphs_knn_dir, graph_name)

            # Now you can use feature_extractor, subfolder, and file_path as needed
            #print("Feature Extractor:", feature_extractor)
            #print("Subfolder:", graphs_subfolder)
            #print("File Path:", file_path)

            # Load graph_data
            graph = torch.load(file_path).to('cuda')
            graph_features = graph["x"].to('cuda')
            graph_coords = graph["centroid"]

            # Assert that the len of the patches with tissue for that ID corresponds with the number of nodes in the graphs
            filtered_imgs_paths = [path for path in selected_images_paths if file_id in path.split("/")[-2]]
            input_shape = (3, 256, 256)

            #case_features = []
            # Read all images paths
            with torch.no_grad():

                if args.context_aware == "NCA":
                    # Aggregate using pretrained attention
                    #Y_hat = model(graph_features)
                    case_aggr_feature_vector = model.milAggregation(graph_features)
                    logits = model.classifier(case_aggr_feature_vector)
                    Y_hat = torch.topk(logits, 1, dim=0)[1]
                    Y_prob = F.softmax(logits, dim=0)

                elif args.context_aware == "CA":
                    Y_prob, Y_hat, logits, case_aggr_feature_vector = model(graph)


                # for img_path in tqdm(filtered_imgs_paths):
                #     img = Image.open(img_path)
                #     img = torch.tensor(np.asarray(img, dtype='uint8'))
                #
                #     #batch_images = images[i:i + batch_size]
                #     img = image_tensor_normalization(x=img, input_shape=input_shape,
                #                                               channel_first=True).to('cuda')
                #     img_features = model.bb(img)
                #     img_features = torch.squeeze(img_features)
                #
                #     case_features.append(img_features)
                #     torch.cuda.empty_cache()

            #case_features_tensor = torch.stack(case_features)

            # Aggregate using pretrained attention
            #case_aggr_feature_vector = model.milAggregation(case_features_tensor)

            #try:
            #print("File:", graph_name)
            #print("Length graph features: ", len(graph_features))
            #print("Length filtered paths: ", len(filtered_imgs_paths))

            #     assert len(graph_features) == len(filtered_paths)
            # except AssertionError:
            #     print("File:", file_name)
            #     print("Length graph features: ", len(graph_features))
            #     print("Length filtered paths: ", len(filtered_paths))
            #     print("Error in feature shape")
            #     continue


            # Find matching label for graph
            id_label = gt_df[gt_df["SUS_number"] == file_id]["MolSubtype_surr"].values[0]

            # Encode task label to numeric value
            encoded_task_label = task_labels_mapping.get(id_label, 0)  # Use 0 as a default value if the label is not found

            # Append features, labels, and task labels
            all_features.append(case_aggr_feature_vector.cpu().detach().numpy())
            all_labels.append(encoded_task_label)
            all_prediction_labels.append(torch.squeeze(Y_hat).cpu().detach().numpy())
            all_task_labels.append(encoded_task_label)

            counter += 1

        # List of tensors 2 single array with predictions
        all_prediction_labels = np.stack(all_prediction_labels)
        all_labels = np.stack(all_labels)

        if fe_taskname == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            class2idx = {0: 'Luminal A', 1: 'Luminal B', 2: 'Her2(+)', 3: 'Triple negative'}
        elif fe_taskname == "LUMINALSvsHER2vsTNBC":
            class2idx = {0: 'Luminal', 1: 'Her2(+)', 2: 'Triple negative'}
        elif fe_taskname == "OTHERvsTNBC":
            class2idx = {0: 'Other', 1: 'Triple negative'}

        clf_report = classification_report(all_labels, all_prediction_labels)
        print(clf_report)
        cfsn_matrix = confusion_matrix(all_labels, all_prediction_labels)

        # Plot
        confusion_matrix_df = pd.DataFrame(cfsn_matrix).rename(columns=class2idx, index=class2idx)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax, cmap='Blues')
        plt.show()


        # Concatenate features for t-SNE
        tsne_features = np.stack(all_features)

        # Apply t-SNE for dimensionality reduction
        tsne_result = TSNE(n_components=2, perplexity=40, n_iter=1000, random_state=42).fit_transform(tsne_features) # , perplexity=5

        # Create a DataFrame for plotting
        #df_tsne = pd.DataFrame(tsne_result, columns=['Dimension 1', 'Dimension 2'])
        # Create a DataFrame for plotting
        df_tsne = pd.DataFrame(
            {'Dimension 1': tsne_result[:, 0], 'Dimension 2': tsne_result[:, 1], 'Labels': all_labels})

        # Plot t-SNE with seaborn
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Dimension 1', y='Dimension 2', hue='Labels', palette='viridis', data=df_tsne) # palette='viridis', sns.color_palette("tab10")
        plt.title(f't-SNE Plot for {args.context_aware} {fe_taskname}')
        plt.show()

        # Compute silhouette score
        silhouette_avg = silhouette_score(tsne_features, all_labels)
        print(f"Silhouette Score for {fe_taskname}: {silhouette_avg}")

        print("\n")


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--path_to_bcnb_dataset', type=str, default="C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB", help="path where h5 files are located")
    parser.add_argument('--path_to_feature_extractors_folder', type=str, default="/Molecular_Subtype_Prediction/data/feature_extractors", help="path where feature extractors are located")
    #parser.add_argument('--tsne_savedir', type=str, default="/output_tsnes", help="path to save graphs folder")
    parser.add_argument('--feats_savedir', type=str, default="../data/extracted_features/NCA_WSI_feats", help="path to save features folder")

    # Feature extractor params
    #parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str, help='Classification task') #
    parser.add_argument("--context_aware", default="NCA", type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument("--graphs_dir", default="../data/CLARIFY/results_graphs_november_23", type=str, help='Directory where graphs are stored.') # "C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB/results_graphs_november_23"
    parser.add_argument("--knn", default=25, type=int, help='KNN used to store graphs.')

    args = parser.parse_args()
    main(args)

print("hola")