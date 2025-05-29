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
from copy import deepcopy
from sklearn.utils import shuffle


def get_optimizer(model, optimizer_type, lr, weight_decay):
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


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
    #plt.show()

    # Close the plot to free up memory
    plt.close(fig)

def custom_categorical_cross_entropy(y_pred, y_true, class_weights=None):
    """
    Computes the categorical cross-entropy loss between the predicted and true class labels.
    """
    loss = torch.nn.CrossEntropyLoss()(y_pred, y_true)
    if class_weights is not None:
        weight_actual_class = class_weights[y_true]
        loss = loss * weight_actual_class
    return loss.mean()


def monte_carlo_cv(X, y, classifier, fe_taskname, n_folds=5, n_repeats=10, batch_size=128,
                            epochs=100, class_weights=None, output_dir='outputs',
                            mlflow_experiment_name="Default", mlflow_server_url=None,
                            lr=0.0001, optimizer_type='adam', owd=None, context_aware='NCA'):
    all_metrics = []

    # Set up MLFlow
    mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    # Create a file to store the indices
    indices_save_path = os.path.join(output_dir, f"train_val_test_indices_{fe_taskname}.txt")
    with open(indices_save_path, 'w') as index_file:

        for repeat in range(n_repeats):
            # Start a nested MLFlow run for each repeat
            with mlflow.start_run(
                    run_name=f"Repeat_{repeat + 1}_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}",
                    nested=True):
                # Log parameters
                mlflow.log_params({
                    "Learning Rate": lr,
                    "Optimizer Type": optimizer_type,
                    "Weight Decay": owd if owd is not None else "None",
                    "Number of Epochs": epochs,
                    "Batch Size": batch_size,
                    "Context Aware": context_aware,
                    "Task": fe_taskname,
                    "N Folds": n_folds,
                    "N Repeats": n_repeats,
                    "Repeat": repeat + 1
                })


                # First split: separate test set (15% of data)
                train_val_idx, test_idx = train_test_split(
                    np.arange(len(y)),
                    test_size=0.15,
                    stratify=y,
                    random_state=42 + repeat
                )

                # Second split: separate train and validation (0.176 of remaining data ≈ 15% of total)
                train_idx, val_idx = train_test_split(
                    train_val_idx,
                    test_size=0.176,
                    stratify=y[train_val_idx],
                    random_state=42 + repeat
                )

                # Log indices for reproducibility
                index_file.write(f"Repeat {repeat + 1}\n")
                index_file.write(f"Train indices: {train_idx.tolist()}\n")
                index_file.write(f"Validation indices: {val_idx.tolist()}\n")
                index_file.write(f"Test indices: {test_idx.tolist()}\n\n")

                # Log indices to MLFlow
                mlflow.log_text(str(train_idx.tolist()), f"Repeat_{repeat + 1}_train_indices.txt")
                mlflow.log_text(str(val_idx.tolist()), f"Repeat_{repeat + 1}_val_indices.txt")
                mlflow.log_text(str(test_idx.tolist()), f"Repeat_{repeat + 1}_test_indices.txt")

                # Create k-fold splits on training data
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42 + repeat)
                fold_metrics = []

                for fold, (fold_train_idx, fold_val_idx) in enumerate(skf.split(X[train_idx], y[train_idx])):
                    print(f"Processing Fold {fold + 1}/{n_folds}")

                    # Map fold indices back to original data indices
                    fold_train_idx = train_idx[fold_train_idx]
                    fold_val_idx = train_idx[fold_val_idx]

                    # Prepare data for training
                    X_train, X_val, X_test = X[fold_train_idx], X[fold_val_idx], X[test_idx]
                    y_train, y_val, y_test = y[fold_train_idx], y[fold_val_idx], y[test_idx]

                    # Convert to PyTorch tensors
                    X_train = torch.tensor(X_train, dtype=torch.float32).to('cuda')
                    y_train = torch.tensor(y_train).to('cuda')
                    X_val = torch.tensor(X_val, dtype=torch.float32).to('cuda')
                    y_val = torch.tensor(y_val).to('cuda')
                    X_test = torch.tensor(X_test, dtype=torch.float32).to('cuda')
                    y_test = torch.tensor(y_test).to('cuda')

                    # Compute class weights if needed
                    if class_weights:
                        classes = np.unique(y_train.cpu().numpy())
                        weights = compute_class_weight('balanced', classes=classes, y=y_train.cpu().numpy())
                        fold_class_weights = torch.tensor(weights, dtype=torch.float32).to('cuda')
                    else:
                        fold_class_weights = None

                    # Initialize model and optimizer
                    current_classifier = deepcopy(classifier).to('cuda')
                    optimizer = get_optimizer(current_classifier, optimizer_type, lr, owd)
                    best_val_loss = float('inf')
                    patience_counter = 0
                    best_model_state = None
                    best_val_confusion_matrix = None
                    best_val_epoch = 0

                    # Training loop
                    for epoch in range(epochs):
                        current_classifier.train()
                        epoch_loss = 0

                        # Train on batches
                        for i in range(0, len(X_train), batch_size):
                            X_batch = X_train[i:i + batch_size]
                            y_batch = y_train[i:i + batch_size]

                            optimizer.zero_grad()
                            logits = current_classifier(X_batch)
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=fold_class_weights)
                            loss.backward()
                            optimizer.step()

                            epoch_loss += loss.item()

                        # Validation phase

                        with torch.no_grad():
                            current_classifier.eval()
                            val_logits = current_classifier(X_val)
                            val_loss = custom_categorical_cross_entropy(val_logits, y_val,
                                                                        class_weights=fold_class_weights)

                            # Calculate validation predictions and confusion matrix
                            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
                            val_true = y_val.cpu().numpy()
                            current_val_cm = confusion_matrix(val_true, val_preds)

                            # Update best model if validation loss improves
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                patience_counter = 0
                                best_model_state = current_classifier.state_dict().copy()
                                best_val_confusion_matrix = current_val_cm
                                best_val_epoch = epoch
                            else:
                                patience_counter += 1

                            if patience_counter >= 20:  # Early stopping threshold
                                print(f"Early stopping triggered at epoch {epoch}")
                                break

                        # Log training metrics
                        mlflow.log_metric(f"Train_Loss_Fold_{fold}", epoch_loss / len(X_train), step=epoch)
                        mlflow.log_metric(f"Val_Loss_Fold_{fold}", val_loss.item(), step=epoch)

                    # Save best validation confusion matrix
                    if best_val_confusion_matrix is not None:
                        val_cm_path = os.path.join(
                            output_dir,
                            f"best_val_confusion_matrix_repeat_{repeat + 1}_fold_{fold + 1}_epoch_{best_val_epoch}.png"
                        )
                        plot_confusion_matrix(best_val_confusion_matrix, np.unique(y), fe_taskname, val_cm_path)
                        mlflow.log_artifact(val_cm_path,
                                            f"validation_confusion_matrices/repeat_{repeat + 1}_fold_{fold + 1}")

                    # Load best model for final evaluation
                    if best_model_state is not None:
                        current_classifier.load_state_dict(best_model_state)

                    # Final evaluation on test set
                    current_classifier.eval()
                    with torch.no_grad():
                        test_logits = current_classifier(X_test)
                        y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
                        y_true = y_test.cpu().numpy()

                        # Calculate final test confusion matrix
                        test_cm = confusion_matrix(y_true, y_pred)
                        test_cm_path = os.path.join(
                            output_dir,
                            f"test_confusion_matrix_repeat_{repeat + 1}_fold_{fold + 1}.png"
                        )
                        plot_confusion_matrix(test_cm, np.unique(y), fe_taskname, test_cm_path)
                        mlflow.log_artifact(test_cm_path,
                                            f"test_confusion_matrices/repeat_{repeat + 1}_fold_{fold + 1}")

                        # Calculate metrics
                        metrics = calculate_metrics(y_true, y_pred, test_logits)

                        # Log metrics
                        for metric_name, value in metrics.items():
                            mlflow.log_metric(f"Test_{metric_name}_Repeat_{repeat + 1}_Fold_{fold + 1}", value)

                        fold_metrics.append({
                            'Repeat': repeat,
                            'Fold': fold,
                            **metrics
                        })

                # After all folds, calculate and log average metrics for this repeat
                fold_metrics_df = pd.DataFrame(fold_metrics)
                avg_fold_metrics = fold_metrics_df.mean()
                for metric_name, value in avg_fold_metrics.items():
                    if metric_name not in ['Repeat', 'Fold']:
                        mlflow.log_metric(f"Avg_{metric_name}_Repeat_{repeat + 1}", value)

                all_metrics.extend(fold_metrics)
                mlflow.end_run()

        # Calculate and log overall metrics
        all_metrics_df = pd.DataFrame(all_metrics)
        overall_metrics = calculate_overall_metrics(all_metrics_df)

        # Final MLFlow run for overall results
        with mlflow.start_run(run_name=f"Overall_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}"):
            log_overall_metrics(overall_metrics, args, fe_taskname=fe_taskname)
            mlflow.end_run()

    return all_metrics_df


def calculate_metrics(y_true, y_pred, logits):
    """Helper function to calculate all metrics"""
    softmax_probs = F.softmax(logits, dim=1).cpu().numpy()

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred, average='weighted'),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted')
    }

    # Add AUC score based on number of classes
    if len(np.unique(y_true)) == 2:
        metrics['AUC'] = roc_auc_score(y_true, softmax_probs[:, 1])
    else:
        metrics['AUC'] = roc_auc_score(y_true, softmax_probs, multi_class='ovr')

    return metrics


def calculate_overall_metrics(metrics_df):
    """Calculate overall metrics across all repeats and folds"""
    metrics_to_average = ['Accuracy', 'F1', 'Precision', 'Recall', 'AUC']
    overall_metrics = {}

    for metric in metrics_to_average:
        overall_metrics[f'mean_{metric}'] = metrics_df[metric].mean()
        overall_metrics[f'std_{metric}'] = metrics_df[metric].std()

    return overall_metrics


def log_overall_metrics(overall_metrics, args, fe_taskname):
    """Log overall metrics and parameters to MLFlow"""
    # Create a dictionary of parameters to log
    params_to_log = {
        "Learning Rate": args.lr,
        "Optimizer Type": args.optimizer_type,
        "Weight Decay": args.owd if args.owd is not None else "None",
        "Epochs": args.epochs,
        "Batch Size": args.batch_size,
        "Context Aware": args.context_aware,
        "Task": fe_taskname,
        "Folds": args.n_folds,
        "Repeats": args.n_repeats
    }

    # Log all collected parameters
    mlflow.log_params(params_to_log)

    # Log metrics
    for metric_name, value in overall_metrics.items():
        mlflow.log_metric(f"Overall_{metric_name}", value)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Leer ground truth y características agregadas desde los grafos
    gt_df = pd.read_excel(args.gt_path)
    graphs_dirs = os.listdir(args.graphs_dir)



    tasks_labels_mappings = {
        "LUMINALAvsLAUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
        "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
        "OTHERvsTNBC": {"Other": 0, "TNBC": 1}
    }

    #Iterate over the graphs

    if args.context_aware == "NCA":
        args.feature_extractors_dir = "../data/feature_extractors"
        feature_extractors = os.listdir(args.feature_extractors_dir)
    elif args.context_aware == "CA":
        args.pretrained_gcn_models_dir = "../data/gcn_pretrained_models/"
        feature_extractors = os.listdir(args.pretrained_gcn_models_dir)
        print("hola")

    # Iterate over the 3 different classification tasks (fe_tasknames)
    for fe_taskname, task_labels_mapping in tasks_labels_mappings.items():
        print(f"Processing task: {fe_taskname}")
        args.fe_task_name = fe_taskname

        all_features, all_labels = [], []

        # Filter graphs for the current task
        for graph_dirname in graphs_dirs:
            # Ensure the graph_dirname matches the current task name
            if fe_taskname not in graph_dirname:
                continue

            try:
                if args.context_aware == "NCA":
                    chosen_model = [fe_name for fe_name in os.listdir(args.feature_extractors_dir) if fe_taskname in fe_name][0]
                    chosen_model_path = os.path.join(args.feature_extractors_dir, chosen_model)
                elif args.context_aware == "CA":
                    if fe_taskname=="LUMINALAvsLAUMINALBvsHER2vsTNBC":
                        fe_taskname="LUMINALAvsLUMINALBvsHER2vsTNBC"
                    chosen_model = [fe_name for fe_name in os.listdir(args.pretrained_gcn_models_dir) if fe_taskname in fe_name][0]
                    args.knn = chosen_model.split("KNN_")[1].split("_")[0]
                    chosen_model_path = os.path.join(args.pretrained_gcn_models_dir, chosen_model)
            except IndexError:
                continue

            print("Chosen model: ", chosen_model)

            # Load the model for this task
            model = torch.load(chosen_model_path).to('cuda')

            # Load graphs for the task
            graphs_knn_dir = os.path.join(args.graphs_dir, graph_dirname, "graphs_k_" + str(args.knn))
            graphs_files = os.listdir(graphs_knn_dir)

            # Extract patient IDs from filenames
            def extract_patient_id(filename):
                match = re.search(r'(SUS\d+)', filename)
                return match.group(1) if match else None

            # Create a DataFrame of graph files and their corresponding SUS numbers
            graph_files_df = pd.DataFrame({
                'filename': graphs_files,
                'SUS_number': [extract_patient_id(filename) for filename in graphs_files]
            })

            # Merge with the ground truth DataFrame to get labels and filter the excluded samples
            merged_df = pd.merge(graph_files_df, gt_df, on='SUS_number', how='inner')
            filtered_df = merged_df[merged_df['Molsub_surr_7clf'] != 'Excluded']

            # For each graph, collect features and labels
            for graph_name in filtered_df['filename'].tolist():
                file_id = graph_name.split("-")[0].split("HE")[0].split("_")[0].split("a")[0]
                file_path = os.path.join(graphs_knn_dir, graph_name)

                # Load the graph
                graph = torch.load(file_path).to('cuda')
                graph_features = graph["x"].to('cuda')

                with torch.no_grad():
                    if args.context_aware == "NCA":
                        case_aggr_feature_vector = model.milAggregation(graph_features)
                    elif args.context_aware == "CA":
                        _, _, _, case_aggr_feature_vector = model(graph)

                # Get the corresponding label for this case
                id_label = gt_df[gt_df["SUS_number"] == file_id]["Molsub_surr_4clf"].values[0]
                encoded_task_label = task_labels_mapping.get(id_label, 0)

                # Add the feature and label for this graph
                all_features.append(case_aggr_feature_vector.cpu().detach().numpy())
                all_labels.append(encoded_task_label)

        # Ensure the data is collected correctly
        assert len(all_features) == len(all_labels) == 534, f"Data size mismatch for task {fe_taskname}!"

        # Convert the features and labels to NumPy arrays
        tsne_features = np.stack(all_features)
        all_labels = np.array(all_labels)

        # Load the pretrained classifier
        classifier = model.classifier

        # Perform Monte Carlo CV and log results
        # Perform Monte Carlo CV and log results
        metrics_df = monte_carlo_cv(
            tsne_features,
            all_labels,
            classifier,
            fe_taskname,
            n_folds=args.n_folds,
            n_repeats=args.n_repeats,
            batch_size=args.batch_size,
            epochs=args.epochs,
            output_dir=args.output_dir,
            mlflow_experiment_name=args.mlflow_experiment_name,
            mlflow_server_url=args.mlflow_server_url,
            lr=args.lr,  # pass the learning rate
            optimizer_type=args.optimizer_type,  # pass the optimizer type
            owd=args.owd,  # pass the weight decay if any
            context_aware=args.context_aware  # pass whether CA or NCA
        )

        # Save metrics
        metrics_output_path = os.path.join(args.output_dir, f"metrics_{fe_taskname}.csv")
        metrics_df.to_csv(metrics_output_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # MLFlow configuration
    parser.add_argument("--mlflow_experiment_name", default="[28052025] HPSearch CA GCN Classifiers new GRAPHS - faircomp", type=str,
                        help='Name for experiment in MLFlow') #[Final] Classifier on Final CBDC 06_09_2024
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # General params
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to save results')
    parser.add_argument('--context_aware', default="CA", type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--n_folds', default=5, type=int, help='Number of folds for Monte Carlo CV')
    parser.add_argument('--n_repeats', default=3, type=int, help='Number of Monte Carlo repeats')
    parser.add_argument('--gt_path', default="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx", type=str, help='Path to ground truth file')
    parser.add_argument('--graphs_dir', default="../data/CLARIFY/results_graphs_january_25", type=str, help='Directory where graphs are stored')
    parser.add_argument('--knn', default=19, type=int, help='KNN used to store graphs')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate of the classifier')
    parser.add_argument('--optimizer_type', default="adam", type=str, help='Optimizer type')
    parser.add_argument('--owd', default=None, type=float, help='Optimizer weight decay')

    # Parse the fixed arguments
    args = parser.parse_args()

    # Lists of hyperparameters to loop over
    lrs = [0.01, 0.001, 0.0001] # [0.01, 0.001, 0.0001, 0.00001, 0.000001] #
    optimizers = ["adam"] #"sgd"
    owds =[0.00001, 0.000001] #[0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [500]
    batch_sizes = [128] #[128, 256]
    context_awareness = ["CA"]


    # Generate all combinations of lrs, optimizers, owds, epochs, and batch_sizes
    for lr, optimizer, owd, epoch, batch_size, context_aware in itertools.product(lrs, optimizers, owds, epochs, batch_sizes,context_awareness):
        # Update the args object with the new hyperparameter values
        args.lr = lr
        args.optimizer_type = optimizer
        args.owd = owd
        args.epochs = epoch
        args.batch_size = batch_size
        args.context_aware = context_aware

        # Print out the current combination (for debugging purposes)
        print(f"Running with lr={lr}, optimizer={optimizer}, owd={owd}, epochs={epoch}, batch_size={batch_size}")

        # Call the main function with the updated args
        main(args)
