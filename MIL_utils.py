import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
import torchvision
import argparse
#import mlflow
import random
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
import datetime
import time
#import seaborn as sns
import sklearn
# import skimage.transform
# import skimage.util
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import cohen_kappa_score, accuracy_score, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

# Stain Normalization
#import torchstain
#import tensorflow as tf
from torchvision import transforms
# import albumentations as A
# from skimage import color
from typing import Optional, Dict
#import yaml

# Graph Convolutional Networks
import torch_geometric
import nmslib
import networkx as nx
from torch_geometric.data import Data as geomData
from itertools import chain
from torch.nn import Linear, LayerNorm, ReLU
from torch_geometric.nn import GCNConv, GraphConv, GatedGraphConv, GATConv, SGConv, GINConv, GENConv, DeepGCNLayer
import igraph as ig
# Models
import torch.nn as nn
import torchvision.models as models
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
#from nystrom_attention import NystromAttention, Nystromformer # https://github.com/lucidrains/nystrom-attention
from collections import defaultdict
import torch.optim as optim

# Old GNN Aggregaion Methods (might delete later)
from torch_geometric.nn import SAGEConv, ClusterGCNConv, global_max_pool, max_pool, dense_diff_pool, DenseSAGEConv

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_train_val_test_ids_dfs(fold_id, files_folds_splitting_ids, dir_cv_dataset_splitting_path, dir_data_frame, pred_column):
    print(f"[CV FOLD {fold_id}]")
    # Retrieve dataset split for each fold
    train_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_train")][0]
    val_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_val")][0]
    test_file = [f for f in files_folds_splitting_ids if f.startswith(f"fold_{fold_id}_test")][0]

    with open(os.path.join(dir_cv_dataset_splitting_path, train_file), "r") as f:
        train_ids = f.read().splitlines()

    with open(os.path.join(dir_cv_dataset_splitting_path, val_file), "r") as f:
        val_ids = f.read().splitlines()

    with open(os.path.join(dir_cv_dataset_splitting_path, test_file), "r") as f:
        test_ids = f.read().splitlines()

    # Convert IDs from strings to ints
    train_ids = [int(id) for id in train_ids]
    val_ids = [int(id) for id in val_ids]
    test_ids = [int(id) for id in test_ids]

    # Read GT DataFrame
    df = pd.read_excel(dir_data_frame)
    df = df[df[pred_column].notna()]  # Clean the rows including NaN values in the column that we want to predict

    # Select df rows by train, test, val split
    train_ids_df = df[df["Patient ID"].isin(train_ids)]  # [:100]#.sample(30)
    val_ids_df = df[df["Patient ID"].isin(val_ids)]  # [:50]#.sample(20)
    test_ids_df = df[df["Patient ID"].isin(test_ids)]  # [:50]#.sample(20)

    return train_ids_df, val_ids_df, test_ids_df

def get_train_test_val_ids(dataset_splitting_path):
    """
    Retrieves the train, test, and validation patient IDs from the BCNB Challenge dataset splitting.

    Parameters:
    dataset_splitting_path (str): The path to the directory containing the train, test, and val splitting files.

    Returns:
    tuple: A tuple containing three lists of patient IDs (train_ids, test_ids, and val_ids).
    """

    # Initialize empty lists to store the train, test, and val patient IDs.
    train_ids = []
    test_ids = []
    val_ids = []

    # Load the train, val, test splitting from the BCNB Challenge
    for root, directories, file in os.walk(dataset_splitting_path):
        for filename in file:
            # If the filename contains "train", open the file and append the patient IDs to the train_ids list.
            if "train" in filename:
                with open(os.path.join(root, filename)) as file:
                    for item in file:
                        train_ids.append(int(item.replace("\n", "")))
            # If the filename contains "test", open the file and append the patient IDs to the test_ids list.
            if "test" in filename:
                with open(os.path.join(root, filename)) as file:
                    for item in file:
                        test_ids.append(int(item.replace("\n", "")))

            # If the filename contains "val", open the file and append the patient IDs to the val_ids list.
            if "val" in filename:
                with open(os.path.join(root, filename)) as file:
                    for item in file:
                        val_ids.append(int(item.replace("\n", "")))

    # Return a tuple containing the train, test, and val patient ID lists.
    return train_ids, test_ids, val_ids

def custom_categorical_cross_entropy(y_pred, y_true, class_weights=None, loss_function="cross_entropy"):
    """
    Computes the categorical cross-entropy loss between the predicted and true class labels.

    Parameters:
    y_pred (torch.Tensor): The predicted class probabilities or logits, with shape (batch_size, num_classes).
    y_true (torch.Tensor): The true class labels, with shape (batch_size,).
    class_weights (torch.Tensor, optional): The class weights, with shape (num_classes,). Default is None.
    loss_function (str, optional): The loss function to use. Must be one of "cross_entropy", "kll", or "mse". Default is "cross_entropy".

    Returns:
    torch.Tensor: The computed loss, with shape (1,).
    """

    # Add a singleton dimension to the predicted class probabilities/logits.
    y_pred = torch.unsqueeze(y_pred, 0)

    # Choose the loss function based on the specified loss_function parameter.
    if loss_function == "cross_entropy":
        loss_not_balanced = torch.nn.CrossEntropyLoss()
    elif loss_function == "kll":
        loss_not_balanced = torch.nn.KLDivLoss()
    elif loss_function == "mse":
        loss_not_balanced = torch.nn.MSELoss()

    # Compute the unweighted loss using the chosen loss function.
    loss = loss_not_balanced(y_pred, y_true)

    # If class weights are specified, compute the weight for the actual class and apply it to the loss.
    if class_weights is not None:
        weight_actual_class = class_weights[y_true]
        loss = loss * weight_actual_class

    # Return the computed loss.
    return loss

def eval_bag_level_classification(test_generator, network, weights2eval_path, pred_column, pred_mode, results_save_path, best_model_type, aggregation, i_epoch, binary=False, return_params=False, show_cf=False, node_feature_extractor=None, knn=None):
    """
    Evaluate bag-level classification performance of the trained neural network.

    Parameters:
    -----------
    test_generator : torch.utils.data.DataLoader
        A DataLoader object containing the test data.
    network : torch.nn.Module
        A neural network model.
    weights2eval_path : str
        The path to the saved weights file.
    pred_column : str
        The column name in the metadata file containing the labels.
    pred_mode : str
        The specific classification task to be performed.
    results_save_path : str
        The path to the directory where the evaluation results will be saved.
    best_model_type : str
        The type of the best performing model.
    aggregation : str
        The type of aggregation method used for the GNN models.
    binary : bool, optional
        Whether the task is a binary classification, by default False.
    return_params : bool, optional
        Whether to return the evaluation parameters, by default False.
    show_cf : bool, optional
        Whether to display the confusion matrix, by default False.
    node_feature_extractor : torch.nn.Module, optional
        A backbone neural network for extracting features from the input images, by default None.
    knn : int, optional
        The number of nearest neighbors, by default None.

    Returns:
    --------
    tuple or None
        If return_params is True, a tuple containing evaluation parameters is returned. Otherwise, None is returned.
    """

    test_network = torch.load(weights2eval_path)
    #network.load_state_dict(test_state_dict)
    test_network.eval()
    print('[EVALUATION / TEST]: at bag level...')

    # Set losses
    # if test_network.mode == 'embedding' or test_network.mode == 'mixed':
    #     L = torch.nn.BCEWithLogitsLoss().cuda()
    # elif test_network.mode == 'instance':
    #     # self.L = torch.nn.BCELoss().cuda()
    #     L = torch.nn.BCELoss().cuda()
        # self.L = torch.nn.CrossEntropyLoss().cuda()

    if aggregation == "GNN" or aggregation == "GNN_basic" or aggregation == "GNN_coords" or aggregation=="Patch":
        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        test_Lce_e = 0
        with torch.no_grad():
            for i_iteration, (X, Y, _, img_coords) in enumerate(tqdm(test_generator, leave=True, position=0)):

                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # Forward network
                if aggregation == "GNN" or aggregation=="GNN_basic" or aggregation=="GNN_coords":
                    Yprob, Yhat, logits, L_gnn = test_network(X, img_coords)
                elif aggregation == "Patch_GCN":
                    # Convert bag to Graph
                    graph_creator = imgs2graph(backbone=node_feature_extractor, pretrained=True, knn=knn).cuda()
                    graph_from_bag = graph_creator(X, img_coords).to('cuda')
                    Yprob, Yhat, logits = test_network(graph=graph_from_bag)
                # Append to list of GTs and preds
                Y_all.append(Y.detach().cpu().numpy())
                Yhat_all.append(Yprob.detach().cpu().numpy().squeeze())
    else: # mean max aggregation
        # Loop over training dataset
        Y_all = []
        Yhat_all = []
        test_Lce_e = 0
        with torch.no_grad():
            for i_iteration, (X, Y, _) in enumerate(tqdm(test_generator, leave=True, position=0)):

                X = torch.tensor(X).cuda().float()
                Y = torch.tensor(Y).cuda().float()

                # Forward network
                Yprob, Yhat, logits = test_network(X)
                # Append to list of GTs and preds
                Y_all.append(Y.detach().cpu().numpy())
                Yhat_all.append(Yprob.detach().cpu().numpy().squeeze())

        # Set class indexes for confusion matrix
        if pred_column == "Molecular subtype":
            if pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                class2idx = {0: 'Luminal A', 1: 'Luminal B', 2: 'Her2(+)', 3: 'Triple negative'}
            elif pred_mode == "LUMINALSvsHER2vsTNBC":
                class2idx = {0: 'Luminal', 1: 'Her2(+)', 2: 'Triple negative'}
            elif pred_mode == "OTHERvsTNBC":
                class2idx = {0: 'Other', 1: 'Triple negative'}

        elif pred_column == "ALN status":
            class2idx = {0: 'N0', 1: 'N+(1-2)', 2: 'N+(>2)'}
        elif pred_column == "HER2 expression":
            class2idx = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        elif pred_column == "Ki67":
            class2idx = {0: 'Low Proliferation (<14%)', 1: 'High Proliferation (>=14%)'}
        elif pred_column == "Histological grading":
            class2idx = {0: '1', 1: '2', 2: '3'}

        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all, axis=1)

        clf_report = classification_report(y_gt, y_pred)
        print(clf_report)

        cfsn_matrix = confusion_matrix(y_gt, y_pred)
        print(cfsn_matrix)

        # Save confusion matrix

        confusion_matrix_df = pd.DataFrame(cfsn_matrix).rename(columns=class2idx, index=class2idx)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax, cmap='Blues')
        cf_savepath = os.path.join(results_save_path, 'test_cfsn_matrix_best_' + str(best_model_type)  + "_" + str(i_epoch)+ '.png')
        plt.savefig(cf_savepath, bbox_inches='tight')
        if show_cf:
            plt.show()

        # # Log CF to mlflow
        # log_cf_every = 1
        # if type(i_epoch) == str: # Training finished
        #     pil_img_cf = Image.frombytes('RGB',
        #                                  fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        #
        # else:
        #     if (i_epoch + 1) % log_cf_every == 0: # Log every log_cf_every epochs
        #         pil_img_cf = Image.frombytes('RGB',
        #                             fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
        #     else:
        #         pil_img_cf = None

        if pred_mode=="OTHERvsTNBC":
        # Calculate Negative Predictive Value (NPV)
            tn, fp, fn, tp = cfsn_matrix.ravel()
            test_npv = tn / (tn + fn)
            if np.isnan((tn / (tn + fn))):
                test_npv = 0

            # Calculate Specificity
            test_specificity = tn / (tn + fp)
            test_tpr = tp / (tp + fn)
            test_tnr = tn / (tn + fp)
            test_fpr = fp / (tn + fp)
            test_fnr = fn / (tp + fn)
        else:
            test_specificity = 0
            test_tpr = 0
            test_tnr = 0
            test_fpr = 0
            test_fnr = 0
            test_npv = 0

        # Compute metrics
        try:
            test_roc_auc_score = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')
        except ValueError as e:
            # Handle the exception here
            print(f"Exception: {e}")
            test_roc_auc_score = 0.

        test_cohen_kappa_score = cohen_kappa_score(y_gt, y_pred)
        test_accuracy_score = accuracy_score(y_gt, y_pred)
        test_f1_score_w = f1_score(y_gt, y_pred, average='weighted')
        test_recall = recall_score(y_gt, y_pred, average='weighted')
        test_precision = precision_score(y_gt, y_pred, average='weighted')
        # Calculate Positive Predictive Value (PPV) or Precision
        test_ppv = precision_score(y_gt, y_pred, average='weighted')

        print("ROC AUC Score ovr: ", test_roc_auc_score)
        print("Cohen Kappa Score: ", test_cohen_kappa_score)
        print("Accuracy Score: ", test_accuracy_score)
        print("F-1 Weighted Score: ", test_f1_score_w)
        print("Recall Score: ", test_recall)
        print("Precision Score: ", test_precision)
        print("Positive Predictive Value (PPV) or Precision:", test_ppv)
        print("Negative Predictive Value (NPV):", test_npv)
        print("Specificity:", test_specificity)
        print("True Positive Rate (TPR) or Recall or Sensitivity:", test_tpr)
        print("True Negative Rate (TNR) or Specificity:", test_tnr)
        print("False Positive Rate (FPR):", test_fpr)
        print("False Negative Rate (FNR):", test_fnr)


        with open(os.path.join(results_save_path, 'test_results_best_' + str(best_model_type) + '.txt'), 'w') as convert_file:
            convert_file.write("\n" +  str(clf_report))
            convert_file.write("\n" + str(cfsn_matrix))
            convert_file.write("\nROC AUC Score ovr: " + str(test_roc_auc_score))
            convert_file.write("\nCohen Kappa Score: " + str(test_cohen_kappa_score))
            convert_file.write("\nAccuracy Score: " + str(test_accuracy_score))
            convert_file.write("\nF-1 Weighted Score: "+ str(test_f1_score_w))
            convert_file.write("\nRecall Score: " + str(test_recall))
            convert_file.write("\nPrecision Score: " + str(test_precision))
            convert_file.write("\nPositive Predictive Value (PPV) or Precision: " + str(test_ppv))
            convert_file.write("\nNegative Predictive Value (NPV): " + str(test_npv))
            convert_file.write("\nSpecificity: " + str(test_specificity))
            convert_file.write("\nTrue Positive Rate (TPR) or Recall or Sensitivity: " + str(test_tpr))
            convert_file.write("\nTrue Negative Rate (TNR) or Specificity: " + str(test_tnr))
            convert_file.write("\nFalse Positive Rate (FPR): " + str(test_fpr))
            convert_file.write("\nFalse Negative Rate (FNR): " + str(test_fnr))

        if return_params:
            return test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath, test_ppv, test_npv, test_specificity, test_tpr, test_tnr, test_fpr, test_fnr

def eval_bag_level_classification_offline_graphs(test_generator, network, weights2eval_path, pred_column, pred_mode, results_save_path, best_model_type, aggregation, return_params=False, show_cf=False):

    """
    Evaluates a PyTorch network for bag level classification task.

    Parameters:
    test_generator (DataLoader): data generator for test data.
    network (nn.Module): instance of PyTorch network to be evaluated.
    weights2eval_path (str): path to weights of the model.
    pred_column (str): column to predict.
    pred_mode (str): prediction mode.
    results_save_path (str): path to save evaluation results.
    best_model_type (str): type of best model to load.
    aggregation (str): type of aggregation to use.
    binary (bool): indicates whether the problem is binary or not (default False).
    return_params (bool): indicates whether to return parameters or not (default False).
    show_cf (bool): indicates whether to show confusion matrix or not (default False).
    node_feature_extractor (str): name of the backbone to be used for feature extraction.
    knn (int): number of neighbors for KNN.

    Returns:
    None. However, evaluation results are printed. Unless return_params is True, which then returns a tuple of the following evaluation metrics:
    - test_roc_auc_score (float): The area under the Receiver Operating Characteristic (ROC) curve.
    - test_cohen_kappa_score (float): The Cohen's Kappa score.
    - test_f1_score_w (float): The weighted F-1 score.
    - test_accuracy_score (float): The classification accuracy score.

    """

    test_network = torch.load(weights2eval_path)
    #network.load_state_dict(test_state_dict)
    test_network.eval()
    print('[EVALUATION / TEST]: at bag level...')

    # Set losses
    # if test_network.mode == 'embedding' or test_network.mode == 'mixed':
    #     L = torch.nn.BCEWithLogitsLoss().cuda()
    # elif test_network.mode == 'instance':
    #     # self.L = torch.nn.BCELoss().cuda()
    #     L = torch.nn.BCELoss().cuda()
        # self.L = torch.nn.CrossEntropyLoss().cuda()
    # Loop over training dataset
    Y_all = []
    Yhat_all = []
    test_Lce_e = 0

    with torch.no_grad():
        for i_iteration, (graph_from_bag, Y) in enumerate(tqdm(test_generator, leave=True, position=0)):
            # Graph to tensor
            graph_from_bag = graph_from_bag.to('cuda')
            # Label to tensor
            Y = torch.tensor(Y).to('cuda')
            #
            #
            # X = torch.tensor(X).cuda().float()
            # Y = torch.tensor(Y).cuda().float()

            # Forward network
            if aggregation == "Patch_GCN_offline":
                Yprob, Yhat, logits = test_network(graph=graph_from_bag)


            # if aggregation == "GNN" or aggregation=="GNN_basic" or aggregation=="GNN_coords":
            #     Yprob, Yhat, logits, L_gnn = test_network(X, img_coords)
            # elif aggregation == "Patch_GCN":
            #     # Convert bag to Graph
            #     graph_creator = imgs2graph(backbone=node_feature_extractor, pretrained=True, knn=knn).cuda()
            #     graph_from_bag = graph_creator(X, img_coords).to('cuda')
            #     Yprob, Yhat, logits = test_network(graph=graph_from_bag)
            # else:
            #     Yprob, Yhat, logits = test_network(X)

            # Yprob, Yhat, logits = test_network(X)

            # Estimate losses
            # Lce = self.L(Yhat, torch.squeeze(Y))
            #test_Lce = L(torch.squeeze(Yhat), torch.squeeze(Y))
            #test_Lce_e += test_Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(Yprob.detach().cpu().numpy().squeeze())

        # Set class indexes for confusion matrix
        if pred_column == "Molecular subtype":
            if pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                class2idx = {0: 'Luminal A', 1: 'Luminal B', 2: 'Her2(+)', 3: 'Triple negative'}
            elif pred_mode == "LUMINALSvsHER2vsTNBC":
                class2idx = {0: 'Luminal', 1: 'Her2(+)', 2: 'Triple negative'}
            elif pred_mode == "OTHERvsTNBC":
                class2idx = {0: 'Other', 1: 'Triple negative'}

        elif pred_column == "ALN status":
            class2idx = {0: 'N0', 1: 'N+(1-2)', 2: 'N+(>2)'}
        elif pred_column == "HER2 expression":
            class2idx = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        elif pred_column == "Ki67":
            class2idx = {0: 'Low Proliferation (<14%)', 1: 'High Proliferation (>=14%)'}
        elif pred_column == "Histological grading":
            class2idx = {0: '1', 1: '2', 2: '3'}

        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all, axis=1)

        clf_report = classification_report(y_gt, y_pred)
        print(clf_report)

        cfsn_matrix = confusion_matrix(y_gt, y_pred)
        print(cfsn_matrix)

        test_roc_auc_score = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')
        test_cohen_kappa_score = cohen_kappa_score(y_gt, y_pred)
        test_accuracy_score = accuracy_score(y_gt, y_pred)
        test_f1_score_w = f1_score(y_gt, y_pred, average='weighted')
        test_recall = recall_score(y_gt, y_pred, average='weighted')
        test_precision = precision_score(y_gt, y_pred, average='weighted')

        print("ROC AUC Score ovr: ", test_roc_auc_score)
        print("Cohen Kappa Score: ", test_cohen_kappa_score)
        print("Accuracy Score: ", test_accuracy_score)
        print("F-1 Weighted Score: ", test_f1_score_w)
        print("Recall Score: ", test_recall)
        print("Precision Score: ", test_precision)

        with open(os.path.join(results_save_path, 'test_results_best_' + str(best_model_type) + '.txt'), 'w') as convert_file:
            convert_file.write("\n" +  str(clf_report))
            convert_file.write("\n" + str(cfsn_matrix))
            convert_file.write("\nROC AUC Score ovr: " + str(test_roc_auc_score))
            convert_file.write("\nCohen Kappa Score: " + str(test_cohen_kappa_score))
            convert_file.write("\nAccuracy Score: " + str(test_accuracy_score))
            convert_file.write("\nF-1 Weighted Score: "+ str(test_f1_score_w))
            convert_file.write("\nRecall Score: " + str(test_recall))
            convert_file.write("\nPrecision Score: " + str(test_precision))


        confusion_matrix_df = pd.DataFrame(cfsn_matrix).rename(columns=class2idx, index=class2idx)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
        cf_savepath = os.path.join(results_save_path, 'test_cfsn_matrix_best_' + str(best_model_type) + '.png')
        plt.savefig(cf_savepath, bbox_inches='tight')
        if show_cf:
            plt.show()

        if return_params:
            return test_roc_auc_score, test_cohen_kappa_score, test_accuracy_score, test_f1_score_w, test_recall, test_precision, cf_savepath

def eval_bag_level_classification_w_return(test_generator, network, weights2eval_path, pred_column, results_save_path, best_model_type, binary=False):
    """
    Evaluates bag-level classification performance of a neural network model using a given test dataset generator.

    Parameters:
    test_generator (torch.utils.data.DataLoader): A PyTorch DataLoader instance representing the test dataset generator.
    network (torch.nn.Module): A PyTorch module representing the neural network model to be evaluated.
    weights2eval_path (str): A string representing the file path to the saved weights of the model to be evaluated.
    pred_column (str): A string representing the column name of the predicted variable in the test dataset.
    results_save_path (str): A string representing the file path to save the evaluation results.
    best_model_type (str): A string representing the type of the best model used for evaluation.
    binary (bool): A boolean representing whether the classification is binary or multiclass.

    Returns:
    A tuple of the following evaluation metrics:
    - test_roc_auc_score (float): The area under the Receiver Operating Characteristic (ROC) curve.
    - test_cohen_kappa_score (float): The Cohen's Kappa score.
    - test_f1_score_w (float): The weighted F-1 score.
    - test_accuracy_score (float): The classification accuracy score.
    """

    # For testing_allmymodels

    test_network = torch.load(weights2eval_path)
    #network.load_state_dict(test_state_dict)
    test_network.eval()
    print('[EVALUATION / TEST]: at bag level...')

    # Set losses
    # if test_network.mode == 'embedding' or test_network.mode == 'mixed':
    #     L = torch.nn.BCEWithLogitsLoss().cuda()
    # elif test_network.mode == 'instance':
    #     # self.L = torch.nn.BCELoss().cuda()
    #     L = torch.nn.BCELoss().cuda()
        # self.L = torch.nn.CrossEntropyLoss().cuda()
    # Loop over training dataset
    Y_all = []
    Yhat_all = []
    test_Lce_e = 0

    with torch.no_grad():
        for i_iteration, (X, Y, _) in enumerate(tqdm(test_generator, leave=True, position=0)):

            X = torch.tensor(X).cuda().float()
            Y = torch.tensor(Y).cuda().float()

            # Forward network
            Yprob, Yhat, logits = test_network(X)

            # Estimate losses
            # Lce = self.L(Yhat, torch.squeeze(Y))
            #test_Lce = L(torch.squeeze(Yhat), torch.squeeze(Y))
            #test_Lce_e += test_Lce.cpu().detach().numpy() / len(test_generator)

            Y_all.append(Y.detach().cpu().numpy())
            Yhat_all.append(Yprob.detach().cpu().numpy().squeeze())


        if pred_column == "Molecular subtype":
            class2idx = {0: 'Luminal A', 1: 'Luminal B', 2: 'Her2(+)', 3: 'Triple negative'}
        elif pred_column == "ALN status":
            class2idx = {0: 'N0', 1: 'N+(1-2)', 2: 'N+(>2)'}
        elif pred_column == "HER2 expression":
            class2idx = {0: '0', 1: '1+', 2: '2+', 3: '3+'}
        elif pred_column == "Ki67":
            class2idx = {0: 'Low Proliferation (<14%)', 1: 'High Proliferation (>=14%)'}
        elif pred_column == "Histological grading":
            class2idx = {0: '1', 1: '2', 2: '3'}

        y_gt = np.argmax(Y_all, axis=1)
        y_pred = np.argmax(Yhat_all, axis=1)

        clf_report = classification_report(y_gt, y_pred)
        #print(clf_report)

        cfsn_matrix = confusion_matrix(y_gt, y_pred)
        #print(cfsn_matrix)

        test_roc_auc_score = roc_auc_score(Y_all, Yhat_all, multi_class='ovr')
        test_cohen_kappa_score = cohen_kappa_score(y_gt, y_pred)
        test_accuracy_score = accuracy_score(y_gt, y_pred)
        test_f1_score_w = f1_score(y_gt, y_pred, average='weighted')

        #print("ROC AUC Score ovr: ", test_roc_auc_score)
        #print("Cohen Kappa Score: ", test_cohen_kappa_score)
        #print("Accuracy Score: ", test_accuracy_score)
        #print("F-1 Weighted Score: ", test_f1_score_w)

        return test_roc_auc_score, test_cohen_kappa_score, test_f1_score_w, test_accuracy_score



# Graph Creation
class imgs2graph(torch.nn.Module):
    """
    A PyTorch module that converts a bag of image patches into a graph using a combination
    of spatial and latent distances.

    Args:
        backbone (str): Name of the feature extractor backbone. Default is 'vgg19'.
        pretrained (bool): Whether to use a pretrained version of the backbone. Default is True.
        knn (int): Number of nearest neighbors to use for constructing the graph. Default is 8.

    Inputs:
        images (torch.Tensor): A batch of image patches.
        img_coords (list): A list of image coordinates.

    Returns:
        G (torch_geometric.data.Data): A graph that combines both spatial and latent edges.
    """

    def __init__(self, backbone='vgg19', pretrained=True, knn=8):
        super(imgs2graph, self).__init__()

        self.backbone = backbone
        self.pretained = pretrained
        self.knn = knn

        # Feature extractor
        self.bb = Encoder( backbone=self.backbone, pretrained=self.pretained)

    def forward(self, images, img_coords):
        # Coordinates of all the patches in the bag
        coords = img_coords

        # Patch-Level feature extraction
        features = self.bb(images)

        # DO EVERYTHING ... create PyTorchGeom data structure
        #graph = "graph in torch geometric form"

        # Convert to np arrays
        coords = np.array(coords)
        features = np.array(features.squeeze().cpu().detach())

        assert coords.shape[0] == features.shape[0]
        num_patches = coords.shape[0]
        radius = self.knn + 1 # we take the K nearest neighbours + 1, because the node itself won't be consider as neighbour

        # Check if instances number in the bag is equal or less than radius - 1
        if num_patches <= radius:
            radius = num_patches

        # Compute spatial distance (based on coordinates)
        model = Hnsw(space='l2')
        model.fit(coords)
        a = np.repeat(range(num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]), # For every coordinate retrieves the indexes of the "spatial"/"coords" nearest neighbours out of all the patches in a bag (based on coords), you take [1:] because the first one corresponds to itself (distance 0), therefore it is actually K=8
                        dtype=int)
        edge_spatial = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Compute latent distance (based on feature vectors)
        model = Hnsw(space='l2')
        model.fit(features)
        a = np.repeat(range(num_patches), radius - 1)
        b = np.fromiter(chain(*[model.query(coords[v_idx], topn=radius)[1:] for v_idx in range(num_patches)]),
                        dtype=int)

        edge_latent = torch.Tensor(np.stack([a, b])).type(torch.LongTensor)

        # Graph that combines both spatial and latent edges
        G = geomData(x=torch.Tensor(features),
                     edge_index=edge_spatial,
                     edge_latent=edge_latent,
                     centroid=torch.Tensor(coords))

        # # Graph based on spatial edges
        # G_spatial = geomData(x=torch.Tensor(features),
        #              edge_index=edge_spatial,
        #              centroid=torch.Tensor(coords))
        # # Graph based on latent edges
        # G_latent = geomData(x=torch.Tensor(features),
        #              edge_index=edge_latent,
        #              centroid=torch.Tensor(coords))
        #
        # #Plot graph
        # g_nx = torch_geometric.utils.to_networkx(G, to_undirected=True)
        # nx.draw(g_nx, cmap=plt.get_cmap('Set3'), node_size=10, linewidths=6)
        # plt.show()

        return G

# Feature extractor for graph creation
class Encoder(torch.nn.Module):
    """Feature extractor for Graph cration.
    Args:
        backbone:
        pretrained:
    Returns:
        features

    """
    def __init__(self, backbone='resnet50_3blocks_1024', pretrained=True):
        super(Encoder, self).__init__()

        self.backbone = backbone
        self.pretrained = pretrained

        if backbone == "resnet50_3blocks_1024":
            resnet = torchvision.models.resnet50(pretrained=True)
            if "3blocks" in backbone:
                self.F = torch.nn.Sequential(resnet.conv1,
                                             resnet.bn1,
                                             resnet.relu,
                                             resnet.maxpool,
                                             resnet.layer1,
                                             resnet.layer2,
                                             resnet.layer3)
            else:
                self.F = resnet

        elif backbone == 'vgg16_512':
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.F = vgg16.features

        elif backbone == 'vgg16_512_BM': # Best model molecular Subtype
            weights2eval_path = "Z:/Shared_PFC-TFG-TFM/Claudio/MIL/MIL_receptors_local/data/results//MolSubtype 512 Clean BDG TP Fixed/PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_ML_10x_NN_bb_vgg16_FBB_True_PS_512_DA_non-spatial_SN_False_L_auc_E_100_LR_0001_AGGR_mean_Order_True_Optim_sgd_N_100_BDG_True_OWD_0.001_TP_O_0.4-T_1-S_1-I_1-N_1/PM_LUMINALAvsLAUMINALBvsHER2vsTNBC_ML_10x_NN_bb_vgg16_FBB_True_PS_512_DA_non-spatial_SN_False_L_auc_E_100_LR_0001_AGGR_mean_Order_True_Optim_sgd_N_100_BDG_True_OWD_0.001_TP_O_0.4-T_1-S_1-I_1-N_1_network_weights_best_auc.pth"
            vgg16_BM = torch.load(weights2eval_path)
            #vgg16_BM.eval()
            self.F = vgg16_BM.bb

    def forward(self, x):
        features = self.F(x)

        features = torch.nn.AdaptiveAvgPool2d((1, 1))(features)

        return features

class Hnsw:
    """
    Approximate Nearest Neighbors search algorithm using Hierarchical Navigable Small World graphs.

    Parameters:
        space (str, optional): The space to index the data points in, defaults to 'cosinesimil'.
        index_params (dict, optional): The index parameters, defaults to None.
        query_params (dict, optional): The query parameters, defaults to None.
        print_progress (bool, optional): Whether to print progress messages, defaults to False.

    Returns:
        Hnsw object: An object of the Hnsw class.
    """

    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=False):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        """
        Fits the Hnsw model to the input data.

        Parameters:
            X (array-like): The input data points to fit the model to.

        Returns:
            Hnsw object: An object of the Hnsw class.
        """

        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn):
        # the knnQuery returns indices and corresponding distance
        # we will throw the distance away for now
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices


#######################
# Stain Normalization #
#######################

class Stain_Normalization():
    """
    Stain normalization is a process of transforming the colors of an image to be similar to a target image.
    This class applies stain normalization using the Macenko method from the torchstain library.

    Parameters:
    -----------
    target_image_path: str
        The file path of the target image used for stain normalization.
    target_shape: tuple
        The size of the target image.

    Returns:
    --------
    numpy.ndarray:
        The normalized image as a numpy array.
    """

    def __init__(self,  target_image_path,target_shape):
        target_image = tf.keras.preprocessing.image.load_img(target_image_path, target_size=target_shape)
        target_image = np.array(target_image)
        target_image = Image.fromarray(target_image, 'RGB')
        torch_normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')

        T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255)
        ])
        torch_normalizer.fit(T(target_image))

        self.transform=T
        self.normalizer=torch_normalizer

    def __call__(self, image, *args, **kwargs):
        image = self.transform(image)
        norm, _, _ = self.normalizer.normalize(I=image, stains=False)
        return norm

# RandStainNA
class RandStainNA(object):
    """
    Class that performs color normalization and augmentation on stained images.

    Parameters:
        yaml_file (str): Path to the YAML file containing the mean and standard deviation of the dataset.
        std_hyper (float, optional): Hyperparameter that controls the range of standard deviation used for the augmentation. Default is 0.
        distribution (str, optional): Type of distribution used to generate the augmented values. It can be "normal", "laplace", or "uniform". Default is "normal".
        probability (float, optional): Probability of applying the augmentation. Default is 1.0.
        is_train (bool, optional): Boolean that indicates if the class is being used for training or not. If True, the input is expected to be a PIL image. If False, the input is expected to be a NumPy array (BGR). Default is True.

    Returns:
        np.ndarray: Augmented image.
    """
    def __init__(
            self,
            yaml_file: str,
            std_hyper: Optional[float] = 0,
            distribution: Optional[str] = 'normal',
            probability: Optional[float] = 1.0,
            is_train: Optional[bool] = True,
    ):

        # true:training setting/false: demo setting

        assert distribution in ['normal', 'laplace', 'uniform'], 'Unsupported distribution style {}.'.format(
            distribution)

        self.yaml_file = yaml_file
        cfg = get_yaml_data(self.yaml_file)
        c_s = cfg['color_space']

        self._channel_avgs = {
            'avg': [cfg[c_s[0]]['avg']['mean'], cfg[c_s[1]]['avg']['mean'], cfg[c_s[2]]['avg']['mean']],
            'std': [cfg[c_s[0]]['avg']['std'], cfg[c_s[1]]['avg']['std'], cfg[c_s[2]]['avg']['std']]
        }
        self._channel_stds = {
            'avg': [cfg[c_s[0]]['std']['mean'], cfg[c_s[1]]['std']['mean'], cfg[c_s[2]]['std']['mean']],
            'std': [cfg[c_s[0]]['std']['std'], cfg[c_s[1]]['std']['std'], cfg[c_s[2]]['std']['std']],
        }

        self.channel_avgs = Dict2Class(self._channel_avgs)
        self.channel_stds = Dict2Class(self._channel_stds)

        self.color_space = cfg['color_space']
        self.p = probability
        self.std_adjust = std_hyper
        self.color_space = c_s
        self.distribution = distribution
        self.is_train = is_train

    def _getavgstd(
            self,
            image: np.ndarray,
            isReturnNumpy: Optional[bool] = True
    ):

        avgs = []
        stds = []

        num_of_channel = image.shape[2]
        for idx in range(num_of_channel):
            avgs.append(np.mean(image[:, :, idx]))
            stds.append(np.std(image[:, :, idx]))

        if isReturnNumpy:
            return (np.array(avgs), np.array(stds))
        else:
            return (avgs, stds)

    def _normalize(
            self,
            img: np.ndarray,
            img_avgs: np.ndarray,
            img_stds: np.ndarray,
            tar_avgs: np.ndarray,
            tar_stds: np.ndarray,
    ) -> np.ndarray:

        img_stds = np.clip(img_stds, 0.0001, 255)
        img = (img - img_avgs) * (tar_stds / img_stds) + tar_avgs

        if self.color_space in ["LAB", "HSV"]:
            img = np.clip(img, 0, 255).astype(np.uint8)

        return img

    def augment(self, img):
        # img:is_train:false——>np.array()(cv2.imread()) #BGR
        # img:is_train:True——>PIL.Image #RGB

        if self.is_train == False:
            image = img
        else:
            image = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        num_of_channel = image.shape[2]

        # color space transfer
        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'HED':
            image = color.rgb2hed(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            )

        std_adjust = self.std_adjust

        # virtual template generation
        tar_avgs = []
        tar_stds = []
        if self.distribution == 'uniform':

            # three-sigma rule for uniform distribution
            for idx in range(num_of_channel):
                tar_avg = np.random.uniform(
                    low=self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                    high=self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                )
                tar_std = np.random.uniform(
                    low=self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                    high=self.channel_avgs.avg[idx] - 3 * self.channel_avgs.std[idx],
                )

                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)
        else:
            if self.distribution == 'normal':
                np_distribution = np.random.normal
            elif self.distribution == 'laplace':
                np_distribution = np.random.laplace

            for idx in range(num_of_channel):
                tar_avg = np_distribution(
                    loc=self.channel_avgs.avg[idx],
                    scale=self.channel_avgs.std[idx] * (1 + std_adjust)
                )

                tar_std = np_distribution(
                    loc=self.channel_stds.avg[idx],
                    scale=self.channel_stds.std[idx] * (1 + std_adjust)
                )
                tar_avgs.append(tar_avg)
                tar_stds.append(tar_std)

        tar_avgs = np.array(tar_avgs)
        tar_stds = np.array(tar_stds)

        img_avgs, img_stds = self._getavgstd(image)

        image = self._normalize(
            img=image,
            img_avgs=img_avgs,
            img_stds=img_stds,
            tar_avgs=tar_avgs,
            tar_stds=tar_stds,
        )

        if self.color_space == 'LAB':
            image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
        elif self.color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        elif self.color_space == 'HED':
            nimg = color.hed2rgb(image)
            imin = nimg.min()
            imax = nimg.max()
            rsimg = (255 * (nimg - imin) / (imax - imin)).astype('uint8')  # rescale to [0,255]

            image = cv2.cvtColor(rsimg, cv2.COLOR_RGB2BGR)

        return image

    def __call__(self, img):
        if np.random.rand(1) < self.p:
            return self.augment(img)
        else:
            return img

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        format_string += f"methods=Reinhard"
        format_string += f", colorspace={self.color_space}"
        format_string += f", mean={self._channel_avgs}"
        format_string += f", std={self._channel_stds}"
        format_string += f", std_adjust={self.std_adjust}"
        format_string += f", distribution={self.distribution}"
        format_string += f", p={self.p})"
        return format_string

def get_yaml_data(yaml_file):

    file = open(yaml_file, 'r', encoding="utf-8")
    file_data = file.read()
    file.close()
    # str->dict
    data = yaml.load(file_data, Loader=yaml.FullLoader)

    return data

class Dict2Class(object):

    def __init__(
            self,
            my_dict: Dict
    ):
        self.my_dict = my_dict
        for key in my_dict:
            setattr(self, key, my_dict[key])