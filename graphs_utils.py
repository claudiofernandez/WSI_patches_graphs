#from MIL_trainer import *
from MIL_models import *
#from MIL_data import *
from MIL_utils import *
import gc


def show_graph_igraph(G):
    G_nx = torch_geometric.utils.to_networkx(G, to_undirected=True)  # convert to NetowrkX Graph

    G_ig = ig.Graph.from_networkx(G_nx)  # Import NX graph

    communities = G_ig.community_edge_betweenness()  # extrat communities
    communities = communities.as_clustering()

    # Plot
    fig1, ax1 = plt.subplots()
    ig.plot(
        communities,
        target=ax1,
        mark_groups=True,
        vertex_size=0.1,
        edge_width=0.5
        # layout=G_ig.layout("rt")
    )
    fig1.set_size_inches(20, 20)

    plt.show()


class WSI2Graph_Generator():

    def __init__(self, pred_column, regions_filled, bag_id, pred_mode, patch_size, data_augmentation,
                 stain_normalization, max_instances, images_on_ram, include_background,
                 balanced_datagen, tissue_percentages_max, feature_extractor_name, num_workers, feature_extractor_dir,
                 model_weights_filename, knn_list, magnification_level,
                 include_edge_features, edges_type, dir_data_frame, dir_dataset_splitting, dir_excels_class_perc,
                 dir_results):

        super(WSI2Graph_Generator, self).__init__()

        # Dataset params
        self.pred_column = pred_column
        self.regions_filled = regions_filled
        self.bag_id = bag_id
        self.pred_mode = pred_mode
        self.patch_size = patch_size
        self.data_augmentation = data_augmentation
        self.max_instances = max_instances
        self.stain_normalization = stain_normalization
        self.images_on_ram = images_on_ram
        self.include_background = include_background
        self.balanced_datagen = balanced_datagen
        self.tissue_percentages_max = tissue_percentages_max
        self.num_workers = num_workers

        # Set classes depending on training task
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.classes = ['Luminal A', 'Luminal B', 'HER2(+)', 'Triple negative']
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.classes = ['Luminal', 'HER2(+)', 'Triple negative']
        elif self.pred_mode == "OTHERvsTNBC":
            self.classes = ['Other', 'Triple negative']

        # Graph parameters
        self.feature_extractor_name = feature_extractor_name
        self.feature_extractor_dir = feature_extractor_dir
        self.feature_extractor_path = os.path.join(feature_extractor_dir, feature_extractor_name)
        self.model_weights_filename = model_weights_filename
        self.knn_list = knn_list
        self.magnification_level = magnification_level  # [BCNB] 5x, 10x, 20x
        self.include_edge_features = include_edge_features  # Include edge_features based on the euclidean distance between the coordinates of each node
        self.edges_type = edges_type  # "latent" based on feature distance or "spatial" based on coordinates distance

        # Change input shape of the images depending on the chosen magnification
        if self.magnification_level == "20x":
            self.input_shape = (3, 512, 512)
        elif self.magnification_level == "10x":
            self.input_shape = (3, 256, 256)
        elif self.magnification_level == "5x":
            self.input_shape = (3, 128, 128)

        # Directories
        self.dir_data_frame = dir_data_frame
        self.dir_dataset_splitting = dir_dataset_splitting
        self.dir_excels_class_perc = dir_excels_class_perc
        self.dir_results = dir_results

        # Determine dir_images
        if self.regions_filled == "fullWSIs_TP_0":
            self.dir_images = "../data/BCNB/preprocessing_results_bien/patches_" + str(
                self.patch_size) + "_fullWSIs_0/"  # '../data/patches_' + str(patch_size) + '_fullWSIs_0/'

        # TODO: Add graph name and save fur definition based on the parameters
        self.graph_name = "graphs_" + self.feature_extractor_name.split("network_weights_best_f1.pth")[0]
        self.dir_results_save_graph = os.path.join(dir_results, self.graph_name)

        # Create dirs for saving resulting graphs
        os.makedirs(self.dir_results, exist_ok=True)
        os.makedirs(self.dir_results_save_graph, exist_ok=True)

        # Obtain train, val and test IDs
        self.train_ids, self.val_ids, self.test_ids = self.get_ids(dir_dataset_splitting=self.dir_dataset_splitting)

        # Obtain GT dfs
        self.train_ids_df, self.val_ids_df, self.test_ids_df = self.get_gt_dfs(train_ids=self.train_ids,
                                                                               val_ids=self.val_ids,
                                                                               test_ids=self.test_ids,
                                                                               dir_data_frame=self.dir_data_frame,
                                                                               pred_column=self.pred_column)
        # Obtain the train, val, and test dataframes conatining the path paths of the split
        self.train_paths_df, self.val_paths_df, self.test_paths_df = self.get_patches_paths_dfs(
            dir_excels_class_perc=self.dir_excels_class_perc)

        # Obtain datasets
        self.dataset_train, self.data_generator_train, self.dataset_val, self.data_generator_val, \
        self.dataset_test, self.data_generator_test = self.get_MIL_datasets_and_dataloaders_coordinates()

        # Generate graphs
        self.create_graphs(knn_list=self.knn_list, data_generator=self.data_generator_train,
                           dir_results_save_graph=self.dir_results_save_graph,
                           feature_extractor_name=self.feature_extractor_name,
                           feature_extractor_path=self.feature_extractor_path,
                           include_edge_features=self.include_edge_features)
        self.create_graphs(knn_list=self.knn_list, data_generator=self.data_generator_val,
                           dir_results_save_graph=self.dir_results_save_graph,
                           feature_extractor_name=self.feature_extractor_name,
                           feature_extractor_path=self.feature_extractor_path,
                           include_edge_features=self.include_edge_features)
        self.create_graphs(knn_list=self.knn_list, data_generator=self.data_generator_test,
                           dir_results_save_graph=self.dir_results_save_graph,
                           feature_extractor_name=self.feature_extractor_name,
                           feature_extractor_path=self.feature_extractor_path,
                           include_edge_features=self.include_edge_features)

    def get_ids(self, dir_dataset_splitting):
        # Dataset split
        train_ids, test_ids, val_ids = get_train_test_val_ids(dir_dataset_splitting)  # function from MIL_data

        return train_ids, val_ids, test_ids

    def get_gt_dfs(self, train_ids, test_ids, val_ids, dir_data_frame, pred_column):

        # Read GT DataFrame
        df = pd.read_excel(dir_data_frame)
        df = df[df[pred_column].notna()]  # Clean the rows including NaN values in the column that we want to predict

        # Select df rows by train, test, val split
        train_ids_df = df[df["Patient ID"].isin(train_ids)]  # .sample(25)
        val_ids_df = df[df["Patient ID"].isin(val_ids)]  # .sample(15)
        test_ids_df = df[df["Patient ID"].isin(test_ids)]  # .sample(15)

        return train_ids_df, val_ids_df, test_ids_df

    def get_patches_paths_dfs(self, dir_excels_class_perc):

        # Read the excel including the images paths and their tissue percentage
        train_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "train_patches_class_perc_0_tp.csv")
        val_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "val_patches_class_perc_0_tp.csv")
        test_class_perc_patches_paths_df = pd.read_csv(dir_excels_class_perc + "test_patches_class_perc_0_tp.csv")

        return train_class_perc_patches_paths_df, val_class_perc_patches_paths_df, test_class_perc_patches_paths_df

    def get_MIL_datasets_and_dataloaders_coordinates(self):

        dataset_train = MILDataset_w_class_perc_coords(dir_images=self.dir_images, data_frame=self.train_ids_df,
                                                       classes=self.classes,
                                                       pred_column=self.pred_column, pred_mode=self.pred_mode,
                                                       magnification_level=self.magnification_level,
                                                       bag_id=self.bag_id, input_shape=self.input_shape,
                                                       data_augmentation=self.data_augmentation,
                                                       stain_normalization=self.stain_normalization,
                                                       images_on_ram=self.images_on_ram,
                                                       include_background=self.include_background,
                                                       class_perc_data_frame=self.train_paths_df,
                                                       tissue_percentages_max=self.tissue_percentages_max)

        data_generator_train = MILDataGenerator_coords_rID(dataset_train, batch_size=1, shuffle=False,
                                                           max_instances=self.max_instances,
                                                           num_workers=self.num_workers,
                                                           pred_column=self.pred_column,
                                                           images_on_ram=self.images_on_ram, pred_mode=self.pred_mode,
                                                           return_patient_id=True)

        # Validation
        dataset_val = MILDataset_w_class_perc_coords(dir_images=self.dir_images, data_frame=self.val_ids_df,
                                                     classes=self.classes,
                                                     pred_column=self.pred_column, pred_mode=self.pred_mode,
                                                     magnification_level=self.magnification_level,
                                                     bag_id=self.bag_id, input_shape=self.input_shape,
                                                     data_augmentation=self.data_augmentation,
                                                     stain_normalization=self.stain_normalization,
                                                     images_on_ram=self.images_on_ram,
                                                     include_background=self.include_background,
                                                     class_perc_data_frame=self.val_paths_df,
                                                     tissue_percentages_max=self.tissue_percentages_max)

        data_generator_val = MILDataGenerator_coords_rID(dataset_val, batch_size=1, shuffle=False,
                                                         max_instances=self.max_instances, num_workers=self.num_workers,
                                                         pred_column=self.pred_column, pred_mode=self.pred_mode,
                                                         images_on_ram=self.images_on_ram, return_patient_id=True)

        # Test
        dataset_test = MILDataset_w_class_perc_coords(dir_images=self.dir_images, data_frame=self.test_ids_df,
                                                      classes=self.classes,
                                                      pred_column=self.pred_column, pred_mode=self.pred_mode,
                                                      magnification_level=self.magnification_level,
                                                      bag_id=self.bag_id, input_shape=self.input_shape,
                                                      data_augmentation=self.data_augmentation,
                                                      stain_normalization=self.stain_normalization,
                                                      images_on_ram=self.images_on_ram,
                                                      include_background=self.include_background,
                                                      class_perc_data_frame=self.test_paths_df,
                                                      tissue_percentages_max=self.tissue_percentages_max)

        data_generator_test = MILDataGenerator_coords_rID(dataset_test, batch_size=1, shuffle=False,
                                                          max_instances=self.max_instances,
                                                          num_workers=self.num_workers, pred_column=self.pred_column,
                                                          pred_mode=self.pred_mode,
                                                          images_on_ram=self.images_on_ram, return_patient_id=True)

        return dataset_train, data_generator_train, dataset_val, data_generator_val, dataset_test, data_generator_test

        print("hola")

    def show_channel_first_patch_from_list(self, list_of_patches, patch_idx):
        plt.imshow(np.transpose(list_of_patches[patch_idx], (1, 2, 0)))
        plt.show()

    def show_graph_igraph(self, G):
        G_nx = torch_geometric.utils.to_networkx(G, to_undirected=True)  # convert to NetowrkX Graph

        G_ig = ig.Graph.from_networkx(G_nx)  # Import NX graph

        communities = G_ig.community_edge_betweenness()  # extrat communities
        communities = communities.as_clustering()

        # Plot
        fig1, ax1 = plt.subplots()
        ig.plot(
            communities,
            target=ax1,
            mark_groups=True,
            vertex_size=0.1,
            edge_width=0.5
            # layout=G_ig.layout("rt")
        )
        fig1.set_size_inches(20, 20)

        plt.show()

    def create_graphs(self, knn_list, data_generator, dir_results_save_graph, feature_extractor_name,
                      feature_extractor_path, include_edge_features):

        # Iterate over K list
        for k in knn_list:
            # Iterate over data generator
            for i_iteration, (patient_id, X, Y, X_augm, img_coords) in enumerate(
                    tqdm(data_generator, leave=True, position=0)):
                graph_savename = patient_id + "_graph.pt"  # "SUS" +  patient_id.zfill(3) + "_graph.pt"
                dir_folder_savegraphs_k = os.path.join(dir_results_save_graph, "graphs_k_" + str(k))
                os.makedirs(dir_folder_savegraphs_k, exist_ok=True)
                if not os.path.isfile(os.path.join(dir_results_save_graph, graph_savename)):
                    if X_augm is None:
                        X_augm = torch.tensor(X)  # .to('cuda')
                    else:
                        X_augm = torch.tensor(X_augm)  # .to('cuda')

                    Y = torch.tensor(Y)  # .to('cuda')

                    # Convert bag to Graph
                    if include_edge_features:
                        graph_creator = imgs2graph_w_edgefeatures_norm(backbone=feature_extractor_name, pretrained=True,
                                                                       knn=k,
                                                                       pretrained_fe_path=feature_extractor_path)  # .cuda()

                    else:
                        graph_creator = imgs2graph(backbone=feature_extractor_name, pretrained=True, knn=k)  # .cuda()

                    graph_from_bag = graph_creator(X_augm, img_coords)  # .to('cuda')

                    # save graphs as .pt files
                    torch.save(graph_from_bag, os.path.join(dir_folder_savegraphs_k, graph_savename))

                    print("hola")


# Feature extractor for graph creation
class Encoder(torch.nn.Module):
    """Feature extractor for Graph cration.
    Args:
        backbone:
        pretrained:
    Returns:
        features

    """

    def __init__(self, backbone='resnet50_3blocks_1024', pretrained=True, pretrained_fe_path=None):
        super(Encoder, self).__init__()

        self.backbone = backbone
        self.pretrained = pretrained
        self.pretrained_fe_path = pretrained_fe_path

        if pretrained_fe_path is not None:  # Pretrained FEs
            if "vgg16" in self.backbone:
                vgg16_BM = torch.load(self.pretrained_fe_path)
                self.F = vgg16_BM.bb
                self.F.to('cuda')
            elif "resnet50" in self.backbone:
                resnet50_BM = torch.load(self.pretrained_fe_path)
                self.F = resnet50_BM.bb
                self.F.to('cuda')

        else:  # Pretrained backbones on imagenet
            if self.backbone == "resnet50_3blocks_1024":
                resnet = torchvision.models.resnet50(pretrained=True)
                if "3blocks" in self.backbone:
                    self.F = torch.nn.Sequential(resnet.conv1,
                                                 resnet.bn1,
                                                 resnet.relu,
                                                 resnet.maxpool,
                                                 resnet.layer1,
                                                 resnet.layer2,
                                                 resnet.layer3)
                else:
                    self.F = resnet
            elif self.backbone == 'vgg16_512':
                vgg16 = torchvision.models.vgg16(pretrained=True)
                self.F = vgg16.features

    def forward(self, x):
        features = self.F(x)

        if self.backbone == "self_sup_histopath":
            return features
        else:
            # features = torch.nn.AdaptiveAvgPool2d((1, 1))(features)
            return features

def image_tensor_normalization(x, input_shape=(3, 512, 512), channel_first=True):
    # Remove the first dimension (assuming it's always 1)
    x = x.squeeze(0)

    # image resize
    x = cv2.resize(np.array(x), (input_shape[1], input_shape[2]))

    # Intensity normalization
    x = x / 255.0
    # Channel first
    if channel_first:
        x = np.transpose(x, (2, 0, 1))

    # Transform back to tensor
    x = torch.tensor(x)
    # Recover first batch dimesnion
    x = x.unsqueeze(0)
    # Numeric type (float32)
    x = x.float()

    return x


class imgs2graph_w_edgefeatures_norm(torch.nn.Module):
    def __init__(self, backbone='vgg19', pretrained=True, knn=8, pretrained_fe_path=None, input_shape=(3, 128, 128)):
        super(imgs2graph_w_edgefeatures_norm, self).__init__()

        self.backbone = backbone
        self.pretrained = pretrained
        self.knn = knn
        self.pretrained_fe_path = pretrained_fe_path
        self.input_shape = input_shape

        # Feature extractor
        if self.pretrained_fe_path is not None:
            self.bb = Encoder(backbone=self.backbone, pretrained=self.pretrained, pretrained_fe_path=self.pretrained_fe_path)
            self.bb.eval()
        else:
            self.bb = Encoder(backbone=self.backbone, pretrained=self.pretrained)
            self.bb.eval()

    def forward(self, images, img_coords, batch_size=1):
        # Coordinates of all the patches in the bag
        coords = img_coords

        # Patch-Level feature extraction
        num_images = images.shape[0]
        print("Extracting features from " + str(num_images) + " images for building WSI-graphs...")
        features = []
        for i in tqdm(range(0, num_images, batch_size)):
            torch.cuda.empty_cache()
            batch_images = images[i:i + batch_size]
            batch_images = image_tensor_normalization(x=batch_images, input_shape=self.input_shape, channel_first=True).to('cuda')
            batch_features = self.bb(batch_images)
            features.append(batch_features)
            del batch_images
            del batch_features
            gc.collect()
            torch.cuda.empty_cache()

        features = torch.cat(features, dim=0)

        # Convert to np arrays
        coords = np.array(coords)
        features = np.array(features.squeeze().cpu().detach())

        assert coords.shape[0] == features.shape[0]
        num_patches = coords.shape[0]
        radius = self.knn + 1

        # Check if instances number in the bag is equal or less than radius - 1
        if num_patches <= radius:
            radius = num_patches

        # Compute spatial distance (based on coordinates)
        model = Hnsw(space='l2')
        model.fit(coords)
        a = np.repeat(range(num_patches), radius - 1)  # shape: [radius - 1 * num_patches]
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),
                        dtype=int)
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Compute latent distance (based on feature vectors)
        model = Hnsw(space='l2')
        model.fit(features)
        a = np.repeat(range(num_patches), radius - 1)  # shape: [radius - 1 * num_patches]
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),
                        dtype=int)
        edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Compute edge_features (based on normalized euclidean distance between coordinates)
        max_coord = np.array([np.max(coords, axis=0)])
        min_coord = np.array([np.min(coords, axis=0)])
        norm_coords = (coords - min_coord) / (max_coord - min_coord)
        edge_features = torch.zeros((edge_spatial.shape[1],))
        for i, (idx1, idx2) in enumerate(edge_spatial.t().tolist()):
            coord1, coord2 = norm_coords[idx1], norm_coords[idx2]
            euclidean_distance = np.sqrt(np.sum((coord1 - coord2) ** 2))
            edge_features[i] = euclidean_distance

        # Graph that combines spatial, latent and euclidean distance as edge features
        G = geomData(x=torch.Tensor(features),
                     edge_index=edge_spatial,
                     edge_latent=edge_latent,
                     edge_features=edge_features,
                     centroid=torch.Tensor(coords))

        return G


if __name__ == '__main__':
    wsi_graph_g = WSI2Graph_Generator(pred_column="Molecular subtype",
                                      regions_filled="fullWSIs_TP_0",
                                      bag_id="Patient ID",
                                      pred_mode="OTHERvsTNBC",
                                      # OTHERvsTNBC,LUMINALSvsHER2vsTNBC, LUMINALAvsLAUMINALBvsHER2vsTNBC
                                      patch_size=512,
                                      data_augmentation=False,  # "non-spatial"
                                      max_instances=420,  # np.inf: all patches from bags
                                      stain_normalization=False,
                                      images_on_ram=False,
                                      include_background=False,
                                      balanced_datagen=False,
                                      tissue_percentages_max="O_0.4-T_1-S_1-I_1-N_1",
                                      num_workers=200,
                                      feature_extractor_name="PM_OTHERvsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10x_N_100_Anetwork_weights_best_f1.pth",
                                      feature_extractor_dir="../output/feature_extractors",
                                      model_weights_filename="network_weights_best_f1.pth",
                                      knn_list=[8, 19, 25],
                                      magnification_level="10x",
                                      include_edge_features=True,
                                      edges_type="spatial",
                                      dir_data_frame="../data/BCNB/patient-clinical-data.xlsx",
                                      dir_dataset_splitting="../data/BCNB/dataset-splitting",
                                      dir_excels_class_perc="../data/BCNB/patches_paths_class_perc/",
                                      dir_results="../output/results_graphs_november_23"
                                      )

    wsi_graph_g = WSI2Graph_Generator(pred_column="Molecular subtype",
                                      regions_filled="fullWSIs_TP_0",
                                      bag_id="Patient ID",
                                      pred_mode="OTHERvsTNBC",
                                      # OTHERvsTNBC,LUMINALSvsHER2vsTNBC, LUMINALAvsLAUMINALBvsHER2vsTNBC
                                      patch_size=512,
                                      data_augmentation=False,  # "non-spatial"
                                      max_instances=420,  # np.inf: all patches from bags
                                      stain_normalization=False,
                                      images_on_ram=False,
                                      include_background=False,
                                      balanced_datagen=False,
                                      tissue_percentages_max="O_0.4-T_1-S_1-I_1-N_1",
                                      num_workers=200,
                                      feature_extractor_name="PM_LUMINALSvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_True_MAGN_10network_weights_best_f1.pth",
                                      feature_extractor_dir="../output/feature_extractors",
                                      model_weights_filename="network_weights_best_f1.pth",
                                      knn_list=[8, 19, 25],
                                      magnification_level="10x",
                                      include_edge_features=True,
                                      edges_type="spatial",
                                      dir_data_frame="../data/BCNB/patient-clinical-data.xlsx",
                                      dir_dataset_splitting="../data/BCNB/dataset-splitting",
                                      dir_excels_class_perc="../data/BCNB/patches_paths_class_perc/",
                                      dir_results="../output/results_graphs_november_23"
                                      )

    wsi_graph_g = WSI2Graph_Generator(pred_column="Molecular subtype",
                                      regions_filled="fullWSIs_TP_0",
                                      bag_id="Patient ID",
                                      pred_mode="OTHERvsTNBC",
                                      # OTHERvsTNBC,LUMINALSvsHER2vsTNBC, LUMINALAvsLAUMINALBvsHER2vsTNBC
                                      patch_size=512,
                                      data_augmentation=False,  # "non-spatial"
                                      max_instances=420,  # np.inf: all patches from bags
                                      stain_normalization=False,
                                      images_on_ram=False,
                                      include_background=False,
                                      balanced_datagen=False,
                                      tissue_percentages_max="O_0.4-T_1-S_1-I_1-N_1",
                                      num_workers=200,
                                      feature_extractor_name="PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_BB_vgg16_AGGR_attention_LR_0.002_OPT_sgd_T_full_dataset_D_BCNB_E_100_L_cross_entropy_OWD_0_FBB_False_PT_Tnetwork_weights_best_f1.pth",
                                      feature_extractor_dir="../output/feature_extractors",
                                      model_weights_filename="network_weights_best_f1.pth",
                                      knn_list=[8, 19, 25],
                                      magnification_level="10x",
                                      include_edge_features=True,
                                      edges_type="spatial",
                                      dir_data_frame="../data/BCNB/patient-clinical-data.xlsx",
                                      dir_dataset_splitting="../data/BCNB/dataset-splitting",
                                      dir_excels_class_perc="../data/BCNB/patches_paths_class_perc/",
                                      dir_results="../output/results_graphs_november_23"
                                      )

    print("hola")


