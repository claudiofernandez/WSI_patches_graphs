from MIL_utils import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import os
import torch
import torch.nn.functional as F
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import re
from collections import Counter
import itertools
from MIL_data import *
from torch.utils.data import Subset, DataLoader
from MIL_models import PatchGCN_MeanMax_LSelec


def plot_confusion_matrix(cm, labels, fe_taskname, cm_image_path):
    if fe_taskname == "LUMINALAvsLUMINALBvsHER2vsTNBC":
        class2idx = {0: 'Luminal A', 1: 'Luminal B', 2: 'Her2(+)', 3: 'TNBC'}
    elif fe_taskname == "LUMINALSvsHER2vsTNBC":
        class2idx = {0: 'Luminal', 1: 'Her2(+)', 2: 'TNBC'}
    elif fe_taskname == "OTHERvsTNBC":
        class2idx = {0: 'Other', 1: 'TNBC'}

    # Plot
    confusion_matrix_df = pd.DataFrame(cm).rename(columns=class2idx, index=class2idx)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax, cmap='Blues')

    plt.title(f'Confusion Matrix - {fe_taskname}')

    # Save the figure to the provided path
    plt.savefig(cm_image_path, bbox_inches='tight')

    # Show the plot
    plt.show()

    # Close the plot to free up memory
    plt.close(fig)


def custom_categorical_cross_entropy(logits, y_true, class_weights=None):
    """
    Computes the categorical cross-entropy loss between the predicted and true class labels.
    """
    loss = torch.nn.CrossEntropyLoss()(logits, y_true.unsqueeze(dim=0))
    if class_weights is not None:
        weight_actual_class = class_weights[y_true]
        loss = loss * weight_actual_class
    return loss.mean()


def monte_carlo_cv(dataset, model_params, fe_taskname, n_folds=5, n_repeats=10, batch_size=128, epochs=100,
                   class_weights=None, output_dir='outputs', mlflow_experiment_name="Default", mlflow_server_url=None,
                   lr=0.0001, optimizer_type='adam', owd=None, context_aware='CA'):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []

    # Set up MLFlow server location, otherwise location of ./mlruns
    if mlflow_server_url:
        mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    # Extract labels from the dataset
    labels = dataset.labels

    # Create a file to store the fold and repeat indices
    indices_save_path = os.path.join(output_dir, f"train_test_indices_{fe_taskname}.txt")
    with open(indices_save_path, 'w') as index_file:

        for repeat in range(n_repeats):
            # Create model with specified architecture
            model = PatchGCN_MeanMax_LSelec(**model_params)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = model.to(device)

            # Start a nested MLFlow run for each repeat
            with mlflow.start_run(
                    run_name=f"Repeat_{repeat + 1}_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}",
                    nested=True):
                # Log parameters once for the run
                mlflow.log_param("Learning Rate", lr)
                mlflow.log_param("Optimizer Type", optimizer_type)
                mlflow.log_param("Weight Decay", owd if owd is not None else "None")
                mlflow.log_param("Number of Epochs", epochs)
                mlflow.log_param("Batch Size", batch_size)
                mlflow.log_param("Context Aware", context_aware)
                mlflow.log_param("Task", fe_taskname)
                mlflow.log_param("Folds", n_folds)
                mlflow.log_param("Repeats", n_repeats)
                mlflow.log_param("GCN Layers", model_params["num_layers"])
                mlflow.log_param("GCN Layer Type", model_params["gnn_layer_type"])
                mlflow.log_param("Graph Pooling", model_params["pooling"])
                mlflow.log_param("Edge Features", model_params["include_edge_features"])
                mlflow.log_param("Dropout", model_params["dropout"])

                fold_metrics = []  # Store metrics for each fold in the current repeat
                cumulative_cm = None  # To store the accumulated confusion matrix

                for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                    print(f"Fold {fold + 1}/{n_folds}")

                    # Log the indices of the training and test set for reproducibility
                    index_file.write(f"Repeat {repeat + 1}, Fold {fold + 1}\n")
                    index_file.write(f"Train indices: {train_index.tolist()}\n")
                    index_file.write(f"Test indices: {test_index.tolist()}\n\n")

                    # Log the train/test indices in MLFlow as well
                    mlflow.log_text(str(train_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_train_indices.txt")
                    mlflow.log_text(str(test_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_test_indices.txt")

                    # Create Subsets for the train and test sets
                    train_subset = Subset(dataset, train_index)
                    test_subset = Subset(dataset, test_index)

                    # Determine prediction column based on task name
                    if fe_taskname == "LUMINALAvsLUMINALBvsHER2vsTNBC":
                        pred_column = "Molsub_surr_4clf"
                    elif fe_taskname == "LUMINALSvsHER2vsTNBC":
                        pred_column = "Molsub_surr_3clf"
                    elif fe_taskname == "OTHERvsTNBC":
                        pred_column = "Molsub_surr_3clf"  # We subclassify between 2 classes later in the Data Generator
                    else:
                        pred_column = "Molsub_surr_4clf"

                    # Initialize data loaders with the custom data generator
                    train_loader = MILDataGenerator_offline_graphs(dataset=train_subset,
                                                                   pred_column=pred_column,
                                                                   pred_mode=fe_taskname,
                                                                   graphs_on_ram=True,
                                                                   shuffle=True,
                                                                   batch_size=batch_size)

                    test_loader = MILDataGenerator_offline_graphs(dataset=test_subset,
                                                                  pred_column=pred_column,
                                                                  pred_mode=fe_taskname,
                                                                  graphs_on_ram=True,
                                                                  shuffle=False,
                                                                  batch_size=batch_size)


                    # Choose optimizer based on the specified optimizer type
                    if optimizer_type == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=owd if owd else 0)
                    elif optimizer_type == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=owd if owd else 0)

                    # Re-train the model for each fold
                    for epoch in tqdm(range(epochs)):
                        model.train()
                        epoch_loss = 0

                        # Training loop using custom data generator for training
                        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                            # Move data to GPU if available
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            X_batch = X_batch.to(device)
                            y_batch = torch.tensor(y_batch).to(device)

                            # Reset the gradients for this batch
                            optimizer.zero_grad()

                            # Forward pass through the model
                            Y_prob, Y_hat, logits, h = model(X_batch)

                            # Calculate the loss using the custom loss function
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=class_weights)

                            # Backpropagate the loss and update the model parameters
                            loss.backward()
                            optimizer.step()

                            # Accumulate the loss for the current epoch
                            epoch_loss += loss.item()

                        # Calculate and log average loss for the epoch
                        avg_epoch_loss = epoch_loss / len(train_loader)
                        mlflow.log_metric(f"Train_Loss_Fold_{fold + 1}", float(np.round(avg_epoch_loss, 4)), step=epoch)

                    # Evaluation on the test set
                    model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    all_y_true = []
                    all_y_pred = []
                    all_logits = []

                    # Testing loop using custom data generator for testing
                    with torch.no_grad():
                        for X_batch, y_batch in tqdm(test_loader):
                            # Move data to GPU if available
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            X_batch = X_batch.to(device)
                            y_batch = torch.tensor(y_batch).to(device)

                            # Forward pass through the model
                            Y_prob, Y_hat, logits, h = model(X_batch)

                            # Calculate the loss on the test set
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=class_weights)
                            test_loss += loss.item()

                            # Calculate accuracy
                            _, predicted = torch.max(logits.data, 1)

                            # Ensure y_batch has a batch dimension if it's a scalar
                            if y_batch.dim() == 0:
                                y_batch = y_batch.unsqueeze(dim=0)

                            # Now you can safely use y_batch in batch-related operations
                            total += y_batch.size(0)
                            correct += (predicted == y_batch).sum().item()

                            # Collect all predictions and true labels
                            all_y_true.extend(y_batch.cpu().numpy())
                            all_y_pred.extend(predicted.cpu().numpy())
                            all_logits.extend(logits)

                    # Log test set metrics
                    avg_test_loss = test_loss / len(test_loader)
                    accuracy = 100 * correct / total
                    mlflow.log_metric(f"Test_Loss_Fold_{fold + 1}", float(np.round(avg_test_loss, 4)))
                    mlflow.log_metric(f"Test_Accuracy_Fold_{fold + 1}", float(np.round(accuracy, 4)))

                    # Calculate additional metrics
                    y_true_labels = np.array(all_y_true)
                    y_pred_labels = np.array(all_y_pred)
                    cm = confusion_matrix(y_true_labels, y_pred_labels)
                    acc = accuracy_score(y_true_labels, y_pred_labels)
                    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')
                    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
                    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')

                    # Convert logits to probabilities
                    all_probs = [F.softmax(logits, dim=0).cpu().numpy() for logits in all_logits]

                    # Flatten the list if needed, or keep it as is if it's a batch
                    all_probs = np.array(all_probs)

                    # Calculate AUC
                    if len(np.unique(y_true_labels)) == 2:
                        auc = roc_auc_score(y_true_labels, np.array(all_probs)[:, 1])
                    else:
                        auc = roc_auc_score(y_true_labels, np.array(all_probs), multi_class='ovr')

                    # Accumulate confusion matrix
                    if cumulative_cm is None:
                        cumulative_cm = cm
                    else:
                        cumulative_cm += cm

                    # Log test metrics
                    mlflow.log_metric(f"Test_F1_Fold_{fold + 1}", f1)
                    mlflow.log_metric(f"Test_Precision_Fold_{fold + 1}", precision)
                    mlflow.log_metric(f"Test_Recall_Fold_{fold + 1}", recall)
                    mlflow.log_metric(f"Test_AUC_Fold_{fold + 1}", auc)

                    fold_metrics.append({
                        'Repeat': repeat + 1,
                        'Fold': fold + 1,
                        'Accuracy': acc,
                        'F1': f1,
                        'Precision': precision,
                        'Recall': recall,
                        'AUC': auc
                    })

                    print(f'Fold {fold + 1}, Epoch [{epochs}/{epochs}], '
                          f'Train Loss: {avg_epoch_loss:.4f}, Test Loss: {avg_test_loss:.4f}, Accuracy: {accuracy:.2f}%')

                # Log the average metrics for this repeat
                fold_metrics_df = pd.DataFrame(fold_metrics)
                avg_fold_metrics = fold_metrics_df.mean()
                mlflow.log_metric(f"Avg_Accuracy_Repeat_{repeat + 1}", avg_fold_metrics['Accuracy'])
                mlflow.log_metric(f"Avg_F1_Repeat_{repeat + 1}", avg_fold_metrics['F1'])
                mlflow.log_metric(f"Avg_Precision_Repeat_{repeat + 1}", avg_fold_metrics['Precision'])
                mlflow.log_metric(f"Avg_Recall_Repeat_{repeat + 1}", avg_fold_metrics['Recall'])
                mlflow.log_metric(f"Avg_AUC_Repeat_{repeat + 1}", avg_fold_metrics['AUC'])

                # Accumulate metrics across repeats
                all_metrics.extend(fold_metrics)

                # Save cumulative confusion matrix after each repeat
                cm_image_path = os.path.join(output_dir, f"cumulative_confusion_matrix_repeat_{repeat + 1}.png")
                plot_confusion_matrix(cumulative_cm, labels=np.unique(labels), fe_taskname=fe_taskname,
                                      cm_image_path=cm_image_path)
                mlflow.log_artifact(cm_image_path, f"confusion_matrices/repeat_{repeat + 1}")

                # End MLFlow run after each repeat
                mlflow.end_run()

        # Calculate overall average metrics across all repeats
        all_metrics_df = pd.DataFrame(all_metrics)
        overall_avg_metrics = all_metrics_df.mean()

        # Log overall average metrics to MLFlow
        mlflow.start_run(
            run_name=f"Overall_Avg_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}")

        mlflow.log_param("Learning Rate", lr)
        mlflow.log_param("Optimizer Type", optimizer_type)
        mlflow.log_param("Weight Decay", owd if owd is not None else "None")
        mlflow.log_param("Number of Epochs", epochs)
        mlflow.log_param("Batch Size", batch_size)
        mlflow.log_param("Context Aware", context_aware)
        mlflow.log_param("Task", fe_taskname)
        mlflow.log_param("Folds", n_folds)
        mlflow.log_param("Repeats", n_repeats)

        mlflow.log_metric("Overall_Avg_Accuracy", overall_avg_metrics['Accuracy'])
        mlflow.log_metric("Overall_Avg_F1", overall_avg_metrics['F1'])
        mlflow.log_metric("Overall_Avg_Precision", overall_avg_metrics['Precision'])
        mlflow.log_metric("Overall_Avg_Recall", overall_avg_metrics['Recall'])
        mlflow.log_metric("Overall_Avg_AUC", overall_avg_metrics['AUC'])
        mlflow.end_run()

    return all_metrics_df, cumulative_cm



import os
import pandas as pd
import itertools
import argparse


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Read ground truth
    gt_df = pd.read_excel(args.gt_path)
    graphs_dirs = os.listdir(args.graphs_dir)

    tasks_labels_mappings = {
        "LUMINALAvsLUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
        "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
        "OTHERvsTNBC": {"Other": 0, "TNBC": 1}
    }

    # Prepare model parameters
    model_params = {
        "dropout": args.drop_out,
        "n_classes": None,  # Will be set based on task
        "num_layers": args.num_gcn_layers,
        "num_features": 512,
        "pooling": args.graph_pooling,
        "include_edge_features": args.include_edge_features,
        "gnn_layer_type": args.gcn_layer_type,
    }

    # Generate hyperparameter combinations once
    param_combinations = list(itertools.product(
        args.lrs, args.optimizers, args.owds, args.epochs_list,
        args.batch_sizes, args.context_awareness,
        args.gcn_layer_types, args.num_gcn_layers_list,
        args.graph_pooling_types, args.knn_values
    ))

    # Iterate over tasks
    for fe_taskname, task_labels_mapping in tasks_labels_mappings.items():
        print(f"Processing task: {fe_taskname}")

        # Set number of classes for current task
        model_params["n_classes"] = len(task_labels_mapping)

        # Filter graphs for current task
        task_graphs = [d for d in graphs_dirs if fe_taskname in d]

        for graph_dirname in task_graphs:
            dataset = MILDataset_offline_graphs(
                args=args,
                graph_dirname=graph_dirname,
                gt_df=gt_df,
                task_labels_mapping=task_labels_mapping,
                graphs_on_ram=True
            )

            # Try each hyperparameter combination
            for (lr, optimizer, owd, epochs, batch_size, context_aware,
                 gcn_layer_type, num_layers, pooling, knn) in param_combinations:
                # Update args with current parameter values
                current_args = argparse.Namespace(**vars(args))
                current_args.lr = lr
                current_args.optimizer_type = optimizer
                current_args.owd = owd
                current_args.epochs = epochs
                current_args.batch_size = batch_size
                current_args.context_aware = context_aware
                current_args.gcn_layer_type = gcn_layer_type
                current_args.num_gcn_layers = num_layers
                current_args.graph_pooling = pooling
                current_args.knn = knn

                # Update model parameters
                current_model_params = model_params.copy()
                current_model_params["gnn_layer_type"] = gcn_layer_type
                current_model_params["num_layers"] = num_layers
                current_model_params["pooling"] = pooling
                #current_model_params["knn"] = knn

                print(f"\nRunning with parameters:")
                print(f"LR: {lr}, Optimizer: {optimizer}, Weight Decay: {owd}")
                print(f"Epochs: {epochs}, Batch Size: {batch_size}")
                print(f"GCN Type: {gcn_layer_type}, Layers: {num_layers}")
                print(f"Pooling: {pooling}, KNN: {knn}")

                # Perform Monte Carlo CV
                metrics_df, cumulative_cm = monte_carlo_cv(
                    dataset=dataset,
                    model_params=current_model_params,
                    fe_taskname=fe_taskname,
                    n_folds=current_args.n_folds,
                    n_repeats=current_args.n_repeats,
                    batch_size=current_args.batch_size,
                    epochs=current_args.epochs,
                    output_dir=current_args.output_dir,
                    mlflow_experiment_name=current_args.mlflow_experiment_name,
                    mlflow_server_url=current_args.mlflow_server_url,
                    lr=current_args.lr,
                    optimizer_type=current_args.optimizer_type,
                    owd=current_args.owd,
                    context_aware=current_args.context_aware
                )

                # Save metrics with comprehensive filename
                metrics_filename = (
                    f"metrics_{fe_taskname}_"
                    f"gcn{gcn_layer_type}_layers{num_layers}_"
                    f"pool{pooling}_knn{knn}_"
                    f"lr{lr}_opt{optimizer}_wd{owd}_"
                    f"bs{batch_size}_ep{epochs}.csv"
                )
                metrics_output_path = os.path.join(args.output_dir, metrics_filename)
                metrics_df.to_csv(metrics_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Original parameters
    parser.add_argument("--mlflow_experiment_name", default="[30112024] Fine-tune GCN on CLARIFY Graphs", type=str,
                        help='Name for experiment in MLFlow')
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to save results')
    parser.add_argument('--context_aware', default="CA", type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--n_folds', default=5, type=int, help='Number of folds for Monte Carlo CV')
    parser.add_argument('--n_repeats', default=2, type=int, help='Number of Monte Carlo repeats')
    parser.add_argument('--gt_path', default="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx", type=str,
                        help='Path to ground truth file')
    parser.add_argument('--graphs_dir', default="../data/CLARIFY/results_graphs_november_23", type=str,
                        help='Directory where graphs are stored')
    parser.add_argument('--knn', default=19, type=int, help='KNN used to store graphs')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate of the classifier')
    parser.add_argument('--optimizer_type', default="adam", type=str, help='Optimizer type')
    parser.add_argument('--owd', default=None, type=float, help='Optimizer weight decay')
    parser.add_argument('--gcn_layer_type', type=str, default="GCNConv", help='Type of GCN layers to use')
    parser.add_argument('--num_gcn_layers', type=int, default=4, help='Number of GCN layers')
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='Graph pooling strategy (mean, max, attention)')
    parser.add_argument('--edge_agg', type=str, default='spatial', help='Edge relationship type (spatial, latent)')
    parser.add_argument('--include_edge_features', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='Include edge features')
    parser.add_argument('--drop_out', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Enable dropout (p=0.25)')

    # Parse arguments
    args = parser.parse_args()

    # Define hyperparameter lists
    args.lrs = [1e-6, 2e-6, 1e-5, 2e-5, 1e-4, 2e-4, 1e-3]
    args.optimizers = ['sgd']
    args.owds = [1e-4, 1e-3, 1e-5]
    args.epochs_list = [200]
    args.batch_sizes = [1]
    args.context_awareness = ['CA']
    args.gcn_layer_types = ['GCNConv', 'GENConv', 'SAGEConv', 'GINConv']
    args.num_gcn_layers_list = [4, 5]
    args.graph_pooling_types = ['mean'] # 'max', 'attention'
    args.knn_values = [8, 19, 25]

    main(args)