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


class MetricsTracker:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_aucs = []
        self.val_aucs = []
        self.train_f1s = []
        self.val_f1s = []
        self.best_val_auc = 0
        self.best_val_f1 = 0
        self.best_epoch = 0

    def update(self, train_metrics, val_metrics):
        self.train_losses.append(train_metrics['loss'])
        self.val_losses.append(val_metrics['loss'])
        self.train_aucs.append(train_metrics['auc'])
        self.val_aucs.append(val_metrics['auc'])
        self.train_f1s.append(train_metrics['f1'])
        self.val_f1s.append(val_metrics['f1'])

        # Update best metrics
        if val_metrics['auc'] > self.best_val_auc:
            self.best_val_auc = val_metrics['auc']
        if val_metrics['f1'] > self.best_val_f1:
            self.best_val_f1 = val_metrics['f1']

    def save_metrics(self, output_dir, repeat):
        metrics_df = pd.DataFrame({
            'train_loss': self.train_losses,
            'val_loss': self.val_losses,
            'train_auc': self.train_aucs,
            'val_auc': self.val_aucs,
            'train_f1': self.train_f1s,
            'val_f1': self.val_f1s
        })
        metrics_df.to_csv(os.path.join(output_dir, f'metrics_repeat_{repeat}.csv'))


def evaluate_metrics_on_loader(model, loader, device, phase='Validation'):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in tqdm(loader, desc=f'Evaluating {phase}'):
            X_batch = X_batch.to(device)
            y_batch = torch.tensor(y_batch).to(device)

            Y_prob, Y_hat, logits, h = model(X_batch)
            loss = custom_categorical_cross_entropy(logits, y_batch)

            total_loss += loss.item()
            all_preds.extend(Y_prob.detach().cpu().numpy())

            # Convert scalar label to one-hot
            n_classes = Y_prob.shape[1]
            label_one_hot = torch.zeros(n_classes)
            label_one_hot[y_batch.cpu()] = 1
            all_labels.append(label_one_hot.numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Calculate metrics
    metrics = {
        'loss': total_loss / len(loader),
        'auc': roc_auc_score(all_labels, all_preds, multi_class='ovr'),
        'f1': f1_score(np.argmax(all_labels, axis=1),
                       np.argmax(all_preds, axis=1),
                       average='weighted'),
        'preds': all_preds,  # Store predictions for confusion matrix
        'labels': all_labels  # Store labels for confusion matrix
    }

    return metrics

def get_optimizer(model, optimizer_type, lr, weight_decay):
    if optimizer_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == "lookahead_adam":
        base_optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return Lookahead(base_optim)
    elif optimizer_type == "lookahead_radam":
        base_optim = torch.optim.RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        return Lookahead(base_optim)


def stratified_kfold_train_val_test(labels, train_size=0.7, val_size=0.15, test_size=0.15, max_tries=1000,
                                    random_seed=42):
    """
    Creates stratified train/val/test splits ensuring all classes are present in each split.

    :param labels: numpy array of shape [N], with integer class labels
    :param train_size: proportion for training set (default 0.7)
    :param val_size: proportion for validation set (default 0.15)
    :param test_size: proportion for test set (default 0.15)
    :return: Tuple of (train_indices, val_indices, test_indices)
    """
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    def has_all_classes(indices):
        return len(np.unique(labels[indices])) == n_classes

    for attempt in range(max_tries):
        # First split: separate test set
        train_val_idx, test_idx = train_test_split(
            np.arange(len(labels)),
            test_size=test_size,
            stratify=labels,
            random_state=random_seed + attempt
        )

        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size_adjusted,
            stratify=labels[train_val_idx],
            random_state=random_seed + attempt
        )

        # Verify all splits have all classes
        if (has_all_classes(train_idx) and
                has_all_classes(val_idx) and
                has_all_classes(test_idx)):
            print(f"Found valid split on attempt {attempt + 1}")
            return train_idx, val_idx, test_idx

    raise ValueError(f"Could not find valid split after {max_tries} attempts")


def stratified_kfold_all_classes(labels, n_splits=3, shuffle=True, max_tries=1000, random_seed=42):
    """
    Attempts to create StratifiedKFold splits such that each fold has *all* classes
    in both train and test splits. Returns a list of (train_index, test_index) if successful.

    :param labels: numpy array of shape [N], with integer class labels.
    :param n_splits: number of folds.
    :param shuffle: whether to shuffle before splitting.
    :param max_tries: how many times to try different random seeds before giving up.
    :param random_seed: base random seed to use for reproducibility.
    :return: A list of (train_index, test_index) pairs.
    :raises ValueError: if we cannot find a split with all classes in each fold within max_tries attempts.
    """
    labels = np.array(labels)
    unique_classes = np.unique(labels)
    n_classes = len(unique_classes)

    # A small helper to check if a given set of indices includes all classes
    def has_all_classes(indices):
        return len(np.unique(labels[indices])) == n_classes

    # Try multiple seeds until we find a valid split
    for attempt in range(max_tries):
        # Use a varying random seed so that we get different shuffles each attempt
        skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_seed + attempt)

        folds = list(skf.split(X=np.zeros(len(labels)), y=labels))
        # Check each fold's train and test to ensure all classes appear
        valid = True
        for (train_index, test_index) in folds:
            # Option A: require each fold's *test set* has all classes
            # Option B: also require each fold's *train set* has all classes
            if not has_all_classes(test_index):
                valid = False
                break
            # If you also need the training set to contain all classes, uncomment:
            # if not has_all_classes(train_index):
            #     valid = False
            #     break

        if valid:
            print(f"Found a valid split on attempt {attempt + 1}")
            return folds

    # If we get here, we couldn't find a valid split
    raise ValueError(f"Could not find a split where every fold contains all classes, "
                     f"even after {max_tries} attempts.")


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
    if class_weights is not None: # Not used when already balancing from the data generator
        weight_actual_class = class_weights[y_true]
        loss = loss * weight_actual_class
    return loss.mean()


def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = torch.tensor(y_batch).to(device)

            Y_prob, Y_hat, logits = model(X_batch)
            all_preds.extend(Y_prob.detach().cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'auc': roc_auc_score(all_labels, all_preds, multi_class='ovr'),
        'f1': f1_score(np.argmax(all_labels, axis=1), np.argmax(all_preds, axis=1), average='weighted'),
        'accuracy': accuracy_score(np.argmax(all_labels, axis=1), np.argmax(all_preds, axis=1))
    }

def get_pred_column(fe_taskname):
    if fe_taskname == "LUMINALAvsLUMINALBvsHER2vsTNBC":
        return "Molsub_surr_4clf"
    elif fe_taskname == "LUMINALSvsHER2vsTNBC":
        return "Molsub_surr_3clf"
    elif fe_taskname == "OTHERvsTNBC":
        return "Molsub_surr_3clf"
    return "Molsub_surr_4clf"


def monte_carlo_cv_with_validation(
        dataset,
        model_params,
        fe_taskname,
        n_repeats=1,
        batch_size=1,
        epochs=100,
        output_dir='outputs',
        mlflow_experiment_name="Default",
        mlflow_server_url=None,
        lr=0.0001,
        optimizer_type='adam',
        optimizer_weight_decay=0.00001,
        early_stopping=True,
        scheduler=True,
        criterion='auc',
        virtual_batch_size=1,
        loss_function='cross_entropy',
        mlflow_log_models=True,
        eval_interval=5  # How often to evaluate on val/test
):
    os.makedirs(output_dir, exist_ok=True)
    if mlflow_server_url:
        mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    all_metrics = []
    labels = np.array(dataset.labels)

    for repeat in range(n_repeats):
        metrics_tracker = MetricsTracker()

        # Create model
        model = PatchGCN_MeanMax_LSelec(**model_params)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        # Get train/val/test splits
        train_idx, val_idx, test_idx = stratified_kfold_train_val_test(
            labels=labels,
            train_size=0.7,
            val_size=0.15,
            test_size=0.15,
            random_seed=42 + repeat
        )

        # Create datasets and loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        test_subset = Subset(dataset, test_idx)

        train_loader = MILDataGenerator_offline_graphs_balanced(
            dataset=train_subset,
            batch_size=batch_size,
            pred_column=get_pred_column(fe_taskname),
            pred_mode=fe_taskname,
            graphs_on_ram=True,
            shuffle=True
        )

        val_loader = MILDataGenerator_offline_graphs(
            dataset=val_subset,
            batch_size=batch_size,
            pred_column=get_pred_column(fe_taskname),
            pred_mode=fe_taskname,
            graphs_on_ram=True,
            shuffle=False
        )

        test_loader = MILDataGenerator_offline_graphs(
            dataset=test_subset,
            batch_size=batch_size,
            pred_column=get_pred_column(fe_taskname),
            pred_mode=fe_taskname,
            graphs_on_ram=True,
            shuffle=False
        )

        # MLFlow run setup
        #run_name = f"Repeat_{repeat + 1}_{fe_taskname}"

        run_name = (
            f"Repeat_{repeat + 1}_{fe_taskname}_"
            f"GCN_{model_params['gnn_layer_type']}_"
            f"Layers{model_params['num_layers']}_"
            f"Pool_{model_params['pooling']}_"
            f"LR_{str(lr).replace('.', '')}_"
            f"Opt_{optimizer_type}"
        )

        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "Learning Rate": lr,
                "Optimizer Type": optimizer_type,
                "Weight Decay": optimizer_weight_decay,
                "Epochs": epochs,
                "Batch Size": batch_size,
                "Virtual Batch Size": virtual_batch_size,
                "Early Stopping": early_stopping,
                "Scheduler": scheduler,
                "Criterion": criterion,
                "Evaluation Interval": eval_interval
            })

            # Initialize optimizer and scheduler
            optimizer = get_optimizer(model, optimizer_type, lr, optimizer_weight_decay)
            if scheduler:
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='max', patience=10, factor=0.5
                )

            best_val_metric = 0
            best_epoch = 0
            patience_counter = 0
            best_model_state = None

            # Training loop
            for epoch in range(epochs):
                # Training phase
                model.train()
                train_loss = 0
                train_preds = []
                train_labels = []

                for batch_idx, (X_batch, y_batch) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}')):
                    X_batch = X_batch.to(device)
                    y_batch = torch.tensor(y_batch).to(device)

                    Y_prob, Y_hat, logits, h = model(X_batch)
                    loss = custom_categorical_cross_entropy(
                        logits,
                        y_batch)
                    #print(h)

                    # Virtual batch handling
                    loss = loss / virtual_batch_size
                    loss.backward()

                    if ((batch_idx + 1) % virtual_batch_size == 0) or (batch_idx + 1 == len(train_loader)):
                        optimizer.step()
                        optimizer.zero_grad()

                    train_loss += loss.item() * virtual_batch_size
                    train_preds.extend(Y_prob.detach().cpu().numpy())
                    # Convert scalar label to one-hot for metrics calculation
                    n_classes = Y_prob.shape[1]
                    label_one_hot = torch.zeros(n_classes)
                    label_one_hot[y_batch.cpu()] = 1
                    train_labels.append(label_one_hot.numpy())

                # Calculate training metrics
                train_metrics = {
                    'loss': train_loss / len(train_loader),
                    'auc': roc_auc_score(train_labels, train_preds, multi_class='ovr'),
                    'f1': f1_score(np.argmax(train_labels, axis=1),
                                   np.argmax(train_preds, axis=1),
                                   average='weighted')
                }

                # Evaluate on validation and test sets every eval_interval epochs
                if epoch % eval_interval == 0:
                    print(f"\nEvaluating at epoch {epoch}")

                    # Validation phase
                    val_metrics = evaluate_metrics_on_loader(model, val_loader, device, 'Validation')

                    # Test phase
                    test_metrics = evaluate_metrics_on_loader(model, test_loader, device, 'Test')

                    # Update learning rate scheduler
                    if scheduler:
                        current_metric = val_metrics['auc'] if criterion == 'auc' else val_metrics['f1']
                        scheduler.step(current_metric)

                    # Model saving logic
                    current_val_metric = val_metrics['auc'] if criterion == 'auc' else val_metrics['f1']
                    if current_val_metric > best_val_metric:
                        best_val_metric = current_val_metric
                        best_epoch = epoch
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()

                        # Save best model
                        model_path = os.path.join(output_dir, f'best_model_repeat_{repeat}.pth')
                        torch.save(model.state_dict(), model_path)
                        if mlflow_log_models:
                            mlflow.log_artifact(model_path)

                        # Save and log confusion matrix
                        val_cm = confusion_matrix(
                            np.argmax(val_metrics['labels'], axis=1),
                            np.argmax(val_metrics['preds'], axis=1)
                        )
                        cm_path = os.path.join(output_dir, f'confusion_matrix_repeat_{repeat}_epoch_{epoch}.png')
                        plot_confusion_matrix(val_cm, labels=np.unique(labels), fe_taskname=fe_taskname,
                                              cm_image_path=cm_path)
                        #plot_confusion_matrix(val_cm, classes=range(model_params['n_classes']),
                        #                      output_path=cm_path)

                        mlflow.log_artifact(cm_path)
                    else:
                        patience_counter += 1

                    # Log metrics
                    mlflow.log_metrics({
                        "train_loss": train_metrics['loss'],
                        "train_auc": train_metrics['auc'],
                        "train_f1": train_metrics['f1'],
                        "val_loss": val_metrics['loss'],
                        "val_auc": val_metrics['auc'],
                        "val_f1": val_metrics['f1'],
                        "test_loss": test_metrics['loss'],
                        "test_auc": test_metrics['auc'],
                        "test_f1": test_metrics['f1'],
                        "best_val_metric": best_val_metric
                    }, step=epoch)

                    # Print metrics
                    print(f"Epoch {epoch}:")
                    print(
                        f"Train - Loss: {train_metrics['loss']:.4f}, AUC: {train_metrics['auc']:.4f}, F1: {train_metrics['f1']:.4f}")
                    print(
                        f"Val   - Loss: {val_metrics['loss']:.4f}, AUC: {val_metrics['auc']:.4f}, F1: {val_metrics['f1']:.4f}")
                    print(
                        f"Test  - Loss: {test_metrics['loss']:.4f}, AUC: {test_metrics['auc']:.4f}, F1: {test_metrics['f1']:.4f}")

                # Early stopping check
                if early_stopping and patience_counter >= 10:
                    print(f'Early stopping triggered at epoch {epoch}')
                    break

                    # After training completion, evaluate best model on test set
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    final_test_metrics = evaluate_metrics_on_loader(model, test_loader, device, 'Final Test')

                    # Log final best metrics
                    all_metrics.append({
                        "repeat": repeat,
                        "best_val_metric": best_val_metric,
                        "best_epoch": best_epoch,
                        "final_test_metrics": final_test_metrics
                    })

                    # Save metrics for this repeat
                    metrics_tracker.save_metrics(output_dir, repeat)

                # End MLFlow run
                mlflow.end_run()

            # Save and return overall results
            results_df = pd.DataFrame(all_metrics)
            results_df.to_csv(os.path.join(output_dir, 'all_results.csv'))
            return results_df


import os
import pandas as pd
import itertools
import argparse


def parse_slurm_arguments():
    parser = argparse.ArgumentParser()

    # MLflow parameters
    parser.add_argument("--mlflow_experiment_name", default="[11012025] Fine-tune GCN on CLARIFY Graphs", type=str,
                        help='Name for experiment in MLFlow')
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # Output and data paths
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to save results')
    parser.add_argument('--gt_path', default="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx", type=str,
                        help='Path to ground truth file')
    parser.add_argument('--graphs_dir', default="../data/CLARIFY/results_graphs_november_23", type=str,
                        help='Directory where graphs are stored')

    # Training parameters
    parser.add_argument('--n_folds', default=2, type=int, help='Number of folds for Monte Carlo CV')
    parser.add_argument('--n_repeats', default=1, type=int, help='Number of Monte Carlo repeats')
    parser.add_argument('--virtual_batch_size', type=int, default=1,
                      help='Virtual batch size for gradient accumulation')
    parser.add_argument('--criterion', default='f1', type=str,
                      help='Criterion for model selection (auc or f1)')
    parser.add_argument('--loss_function', default='cross_entropy', type=str,
                      help='Loss function to use')
    parser.add_argument('--mlflow_log_models', default=True,
                      type=lambda x: (str(x).lower() == 'true'),
                      help='Whether to log models to MLFlow')


    # Model hyperparameters (previously in lists, now individual)
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--optimizer_type', type=str, required=True, choices=['adam', 'sgd'], help='Optimizer type')
    parser.add_argument('--owd', type=float, required=True, help='Optimizer weight decay')
    parser.add_argument('--epochs', type=int, default=25, required=False, help='Number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, required=False, help='Batch size')
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

            # Perform Monte Carlo CV with validation
            metrics_df = monte_carlo_cv_with_validation(
                dataset=dataset,
                model_params=model_params,
                fe_taskname=fe_taskname,
                n_repeats=args.n_repeats,
                batch_size=1,  # Using batch_size=1 as in original implementation
                epochs=args.epochs,
                output_dir=args.output_dir,
                mlflow_experiment_name=args.mlflow_experiment_name,
                mlflow_server_url=args.mlflow_server_url,
                lr=args.lr,
                optimizer_type=args.optimizer_type,
                optimizer_weight_decay=args.owd,  # Changed from owd to optimizer_weight_decay
                early_stopping=True,
                scheduler=True,
                criterion=args.criterion,
                virtual_batch_size=args.virtual_batch_size,
                loss_function=args.loss_function,
                mlflow_log_models=args.mlflow_log_models
            )

            # Save metrics with comprehensive filename
            metrics_filename = (
                f"metrics_{fe_taskname}_"
                f"gcn{args.gcn_layer_type}_layers{args.num_gcn_layers}_"
                f"pool{args.graph_pooling}_knn{args.knn}_"
                f"lr{args.lr}_opt{args.optimizer_type}_wd{args.owd}_"
                f"vbs{args.virtual_batch_size}_"  # Added virtual batch size
                f"ep{args.epochs}.csv"
            )
            metrics_output_path = os.path.join(args.output_dir, metrics_filename)
            metrics_df.to_csv(metrics_output_path, index=False)

            # Log experiment completion
            print(f"Completed experiment for {fe_taskname} with configuration:")
            print(f"- GCN Type: {args.gcn_layer_type}")
            print(f"- Layers: {args.num_gcn_layers}")
            print(f"- Pooling: {args.graph_pooling}")
            print(f"- Learning Rate: {args.lr}")
            print(f"- Optimizer: {args.optimizer_type}")
            print(f"- Weight Decay: {args.owd}")
            print(f"- Virtual Batch Size: {args.virtual_batch_size}")
            print(f"Results saved to: {metrics_output_path}")

if __name__ == "__main__":
    main()