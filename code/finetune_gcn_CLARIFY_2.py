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


def monte_carlo_cv(dataset, model, fe_taskname, n_folds=5, n_repeats=10, batch_size=128, epochs=100,
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

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Leer ground truth y características agregadas desde los grafos
    gt_df = pd.read_excel(args.gt_path)
    graphs_dirs = os.listdir(args.graphs_dir)

    tasks_labels_mappings = {
        "LUMINALAvsLUMINALBvsHER2vsTNBC": {"Luminal A": 0, "Luminal B": 1, "HER2(+)": 2, "TNBC": 3},
        "LUMINALSvsHER2vsTNBC": {"Luminal": 0, "HER2(+)": 1, "TNBC": 2},
        "OTHERvsTNBC": {"Other": 0, "TNBC": 1}
    }

    # Iterate over the graphs

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

        all_features, all_labels = [], []

        # Filter graphs for the current task
        for graph_dirname in graphs_dirs:
            # Ensure the graph_dirname matches the current task name
            if fe_taskname not in graph_dirname:
                continue

            try:
                if args.context_aware == "NCA":
                    chosen_model = \
                    [fe_name for fe_name in os.listdir(args.feature_extractors_dir) if fe_taskname in fe_name][0]
                    chosen_model_path = os.path.join(args.feature_extractors_dir, chosen_model)
                elif args.context_aware == "CA":
                    chosen_model = \
                    [fe_name for fe_name in os.listdir(args.pretrained_gcn_models_dir) if fe_taskname in fe_name][0]
                    args.knn = chosen_model.split("KNN_")[1].split("_")[0]
                    chosen_model_path = os.path.join(args.pretrained_gcn_models_dir, chosen_model)
            except IndexError:
                continue

            print("Chosen model: ", chosen_model)

            # Load the model for this task
            model = torch.load(chosen_model_path).to('cuda')

            dataset = MILDataset_offline_graphs(args=args, graph_dirname=graph_dirname,
                                                gt_df=gt_df, task_labels_mapping=task_labels_mapping,
                                                graphs_on_ram=True)

            # Perform Monte Carlo CV and log results
            metrics_df, cumulative_cm = monte_carlo_cv(
                dataset,
                model,
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
    parser.add_argument("--mlflow_experiment_name", default="[12102024] Fine-tune GCN on CLARIFY Graphs", type=str,
                        help='Name for experiment in MLFlow')  # [Final] Classifier on Final CBDC 06_09_2024
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # General params
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

    # Parse the fixed arguments
    args = parser.parse_args()

    # Lists of hyperparameters to loop over
    lrs = [2e-4, 1e-4]
    optimizers = ['adam'] #, 'sgd']
    owds = [1e-05]
    epochs = [100]
    batch_sizes = [1]
    context_awareness = ['CA']

    # Generate all combinations of lrs, optimizers, owds, epochs, and batch_sizes
    for lr, optimizer, owd, epoch, batch_size, context_aware in itertools.product(lrs, optimizers, owds, epochs,
                                                                                  batch_sizes, context_awareness):
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
