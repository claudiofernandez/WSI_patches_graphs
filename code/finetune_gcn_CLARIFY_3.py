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


def monte_carlo_cv(dataset, model_params, fe_taskname, n_folds=3, n_repeats=1, batch_size=128, epochs=100,
                   class_weights=None, output_dir='outputs', mlflow_experiment_name="Default", mlflow_server_url=None,
                   lr=0.0001, optimizer_type='adam', owd=None, context_aware='CA', patience=30):
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
                # Log parameters
                mlflow.log_param("Learning Rate", lr)
                mlflow.log_param("Optimizer Type", optimizer_type)
                mlflow.log_param("Weight Decay", owd if owd is not None else "None")
                mlflow.log_param("Number of Epochs", epochs)
                mlflow.log_param("Batch Size", batch_size)
                mlflow.log_param("Early Stopping Patience", patience)
                mlflow.log_param("Context Aware", context_aware)
                mlflow.log_param("Task", fe_taskname)
                mlflow.log_param("Folds", n_folds)
                mlflow.log_param("Repeats", n_repeats)
                mlflow.log_param("GCN Layers", model_params["num_layers"])
                mlflow.log_param("GCN Layer Type", model_params["gnn_layer_type"])
                mlflow.log_param("Graph Pooling", model_params["pooling"])
                mlflow.log_param("Edge Features", model_params["include_edge_features"])
                mlflow.log_param("Dropout", model_params["dropout"])

                fold_metrics = []
                cumulative_cm = None

                for fold, (train_index, test_index) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                    print(f"Fold {fold + 1}/{n_folds}")

                    # Log indices
                    index_file.write(f"Repeat {repeat + 1}, Fold {fold + 1}\n")
                    index_file.write(f"Train indices: {train_index.tolist()}\n")
                    index_file.write(f"Test indices: {test_index.tolist()}\n\n")

                    mlflow.log_text(str(train_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_train_indices.txt")
                    mlflow.log_text(str(test_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_test_indices.txt")

                    # Create Subsets
                    train_subset = Subset(dataset, train_index)
                    test_subset = Subset(dataset, test_index)

                    # Determine prediction column
                    if fe_taskname == "LUMINALAvsLUMINALBvsHER2vsTNBC":
                        pred_column = "Molsub_surr_4clf"
                    elif fe_taskname == "LUMINALSvsHER2vsTNBC":
                        pred_column = "Molsub_surr_3clf"
                    elif fe_taskname == "OTHERvsTNBC":
                        pred_column = "Molsub_surr_3clf"
                    else:
                        pred_column = "Molsub_surr_4clf"

                    # Initialize data loaders
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

                    # Choose optimizer
                    if optimizer_type == "adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=owd if owd else 0)
                    elif optimizer_type == "sgd":
                        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=owd if owd else 0)

                    # Early stopping variables
                    best_loss = float('inf')
                    patience_counter = 0
                    best_model_state = None
                    stopped_epoch = epochs

                    # Training loop
                    for epoch in tqdm(range(epochs)):
                        model.train()
                        epoch_loss = 0

                        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            X_batch = X_batch.to(device)
                            y_batch = torch.tensor(y_batch).to(device)

                            optimizer.zero_grad()
                            Y_prob, Y_hat, logits, h = model(X_batch)
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=class_weights)
                            loss.backward()
                            optimizer.step()

                            epoch_loss += loss.item()

                        avg_epoch_loss = epoch_loss / len(train_loader)
                        mlflow.log_metric(f"Train_Loss_Fold_{fold + 1}", float(np.round(avg_epoch_loss, 4)), step=epoch)

                        # Early stopping check
                        if avg_epoch_loss < best_loss:
                            best_loss = avg_epoch_loss
                            patience_counter = 0
                            best_model_state = model.state_dict().copy()
                        else:
                            patience_counter += 1

                        if patience_counter >= patience:
                            print(f'Early stopping triggered at epoch {epoch + 1}')
                            stopped_epoch = epoch + 1
                            model.load_state_dict(best_model_state)
                            break

                    # Log early stopping epoch
                    mlflow.log_metric(f"Stopped_Epoch_Fold_{fold + 1}", stopped_epoch)

                    # Evaluation
                    model.eval()
                    test_loss = 0
                    correct = 0
                    total = 0
                    all_y_true = []
                    all_y_pred = []
                    all_logits = []

                    with torch.no_grad():
                        for X_batch, y_batch in tqdm(test_loader):
                            device = 'cuda' if torch.cuda.is_available() else 'cpu'
                            X_batch = X_batch.to(device)
                            y_batch = torch.tensor(y_batch).to(device)

                            Y_prob, Y_hat, logits, h = model(X_batch)
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=class_weights)
                            test_loss += loss.item()

                            _, predicted = torch.max(logits.data, 1)

                            if y_batch.dim() == 0:
                                y_batch = y_batch.unsqueeze(dim=0)

                            total += y_batch.size(0)
                            correct += (predicted == y_batch).sum().item()

                            all_y_true.extend(y_batch.cpu().numpy())
                            all_y_pred.extend(predicted.cpu().numpy())
                            all_logits.extend(logits)

                    # Calculate and log metrics
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


def parse_slurm_arguments():
    parser = argparse.ArgumentParser()

    # MLflow parameters
    parser.add_argument("--mlflow_experiment_name", default="[30112024] Fine-tune GCN on CLARIFY Graphs", type=str,
                        help='Name for experiment in MLFlow')
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # Output and data paths
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to save results')
    parser.add_argument('--gt_path', default="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx", type=str,
                        help='Path to ground truth file')
    parser.add_argument('--graphs_dir', default="../data/CLARIFY/results_graphs_november_23", type=str,
                        help='Directory where graphs are stored')

    # Training parameters
    parser.add_argument('--n_folds', default=3, type=int, help='Number of folds for Monte Carlo CV')
    parser.add_argument('--n_repeats', default=1, type=int, help='Number of Monte Carlo repeats')

    # Model hyperparameters (previously in lists, now individual)
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--optimizer_type', type=str, required=True, choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--owd', type=float, required=True, help='Optimizer weight decay')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--context_aware', type=str, required=True, choices=['CA', 'NCA'],
                        help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument('--gcn_layer_type', type=str, required=True,
                        choices=['GCNConv', 'GENConv', 'SAGEConv', 'GINConv'],
                        help='Type of GCN layers')
    parser.add_argument('--num_gcn_layers', type=int, required=True, help='Number of GCN layers')
    parser.add_argument('--graph_pooling', type=str, required=True, choices=['attention', 'mean', 'max'],
                        help='Graph pooling strategy')
    parser.add_argument('--knn', type=int, required=True, help='KNN value')

    # Other model parameters
    parser.add_argument('--edge_agg', type=str, default='spatial', help='Edge relationship type (spatial, latent)')
    parser.add_argument('--include_edge_features', default=False,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Include edge features')
    parser.add_argument('--drop_out', default=True,
                        type=lambda x: (str(x).lower() == 'true'),
                        help='Enable dropout (p=0.25)')

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_slurm_arguments()
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

            # Perform Monte Carlo CV
            metrics_df, cumulative_cm = monte_carlo_cv(
                dataset=dataset,
                model_params=model_params,
                fe_taskname=fe_taskname,
                n_folds=args.n_folds,
                n_repeats=args.n_repeats,
                batch_size=args.batch_size,
                epochs=args.epochs,
                output_dir=args.output_dir,
                mlflow_experiment_name=args.mlflow_experiment_name,
                mlflow_server_url=args.mlflow_server_url,
                lr=args.lr,
                optimizer_type=args.optimizer_type,
                owd=args.owd,
                context_aware=args.context_aware
            )

            # Save metrics with comprehensive filename
            metrics_filename = (
                f"metrics_{fe_taskname}_"
                f"gcn{args.gcn_layer_type}_layers{args.num_gcn_layers}_"
                f"pool{args.graph_pooling}_knn{args.knn}_"
                f"lr{args.lr}_opt{args.optimizer_type}_wd{args.owd}_"
                f"bs{args.batch_size}_ep{args.epochs}.csv"
            )
            metrics_output_path = os.path.join(args.output_dir, metrics_filename)
            metrics_df.to_csv(metrics_output_path, index=False)


if __name__ == "__main__":
    main()