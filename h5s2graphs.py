import h5py
import time
import os
from tqdm import tqdm
import argparse
import torch
from graphs_utils import *


def read_assets_from_h5(h5_path):
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            assets[key] = f[key][:]
            if f[key].attrs is not None:
                attrs[key] = dict(f[key].attrs)
    return assets, attrs

def create_graph(knn_list, wsi_assets, wsi_attr, dir_results_save_graph, feature_extractor_name, feature_extractor_path, include_edge_features, input_shape):

    # Extract patches imgs and coords data
    images = torch.tensor(wsi_assets["imgs"])
    coords = np.round(wsi_assets["coords"] / 512).astype(int) #Divide over patch size (512) to adjust to previous patching method

    # Extract image name
    wsi_name = wsi_attr["imgs"]["wsi_name"]
    graph_savename = wsi_name + "_graph.pt"  # "SUS" +  patient_id.zfill(3) + "_graph.pt"

    # Iterate over K list
    for k in knn_list:
        dir_folder_savegraphs_k = os.path.join(dir_results_save_graph, "graphs_k_" + str(k))
        os.makedirs(dir_folder_savegraphs_k, exist_ok=True)

        if not os.path.isfile(os.path.join(dir_results_save_graph, graph_savename)):
            # Convert bag to Graph
            if include_edge_features:
                graph_creator = imgs2graph_w_edgefeatures_norm(backbone=feature_extractor_name, pretrained=True,
                                                               knn=k,
                                                               pretrained_fe_path=feature_extractor_path, input_shape=input_shape)  # .cuda()

            graph_from_bag = graph_creator(images=images, img_coords=coords, batch_size=1)  # .to('cuda')

            # save graphs as .pt files
            torch.save(graph_from_bag, os.path.join(dir_folder_savegraphs_k, graph_savename))

            print("hola")


def main(args):

    # Set up directories depending on where this program is executed
    if args.where_exec == "slurm_nas":
        dir_results_save_graph = os.path.join('/workspace/NASFolder', args.graph_savedir)
        dir_h5s = os.path.join('/workspace/NASFolder', args.path_to_h5files)
        feature_extractor_dir = os.path.join('/workspace/NASFolder', args.path_to_feature_extractors_folder)
    elif args.where_exec == "slurm_dgx":
        dir_results_save_graph = os.path.join('/workspace/NASFolder', args.graph_savedir)
        dir_h5s = os.path.join('/workspace/DGXFolder', args.path_to_h5files)
        feature_extractor_dir = os.path.join('/workspace/DGXFolder', args.path_to_feature_extractors_folder)
    elif args.where_exec == "dgx_gpu":
        dir_results_save_graph = os.path.join('/workspace/exec/NASFolder', args.graph_savedir)
        dir_h5s = os.path.join("/workspace/exec/dataDGX/", args.path_to_h5files)
        feature_extractor_dir = os.path.join('/workspace/exec/NASFolder', args.path_to_feature_extractors_folder)
    elif args.where_exec == "local":
        dir_results_save_graph = os.path.join('F:', args.graph_savedir)
        dir_h5s = os.path.join('F:', args.path_to_h5files)
        feature_extractor_dir = os.path.join('C:/Users/clferma1/Documents/Investigacion_GIT', args.path_to_feature_extractors_folder)


    print(dir_results_save_graph, dir_h5s, feature_extractor_dir)

    os.makedirs(dir_results_save_graph, exist_ok=True) # create output directory

    # Specify the path to your HDF5 file
    h5_files = os.listdir(dir_h5s)

    # Read h5 files
    for h5file in tqdm(h5_files):
        # Derive filepath
        file_path = os.path.join(dir_h5s, h5file)

        # Directories
        feature_extractor_path = os.path.join(feature_extractor_dir, args.feature_extractor_name)

        # Record the start time
        print(f"Extracting information from h5 file " + h5file + " ...")
        start_time = time.time()
        # Read h5 file
        assets, attr = read_assets_from_h5(file_path)
        # Record the end time
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Elapsed time reading h5: {elapsed_time} seconds")

        # Generate graphs
        create_graph(knn_list=args.knn_list, wsi_assets=assets, wsi_attr=attr,
                    dir_results_save_graph=dir_results_save_graph,
                    feature_extractor_name=args.feature_extractor_name,
                    feature_extractor_path=feature_extractor_path,
                    include_edge_features=args.include_edge_features,
                    input_shape=args.input_shape)


if __name__ == '__main__':

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--where_exec', type=str, default="dgx_gpu", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--path_to_h5files', type=str, default="CLAUDIO/BREAST_CANCER_DATASETS/CLARIFY_BREAST_CANCER_DATABASE_NOV2023/Results_CLAM/h5s/patches", help="path where h5 files are located")
    parser.add_argument('--path_to_feature_extractors_folder', type=str, default="Molecular_Subtype_Prediction/data/feature_extractors", help="path where feature extractors are located")
    parser.add_argument('--graph_savedir', type=str, default="output_graphs", help="path to save graphs folder")

    # Feature extractor params
    parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str, help='Classification task')
    parser.add_argument("--feature_extractor_name", default="PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10x_N_100_Anetwork_weights_best_f1.pth", type=str, help='Chosen feature extractor.')

    # Graph params
    parser.add_argument("--knn_list", default=[8, 19, 25], type=list, help='KNN values for generating graphs')
    parser.add_argument('--include_edge_features', default=True, type=lambda x: (str(x).lower() == 'true'), help="Include edge features.")
    parser.add_argument("--edges_type", default="spatial", type=str, help='Type of edge: spatial/latent')
    parser.add_argument("--input_shape", default=(3, 256, 256), type=tuple, help='Input shape for the feature extractor.')

    args = parser.parse_args()
    main(args)

print("hola")