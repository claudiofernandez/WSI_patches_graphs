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


def plot_confusion_matrix(cm, labels, fe_taskname, cm_image_path):

    if fe_taskname == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
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


def monte_carlo_cv(X, y, classifier, fe_taskname, n_folds=5, n_repeats=10, batch_size=128, epochs=100, class_weights=None,
                   output_dir='outputs', mlflow_experiment_name="Default", mlflow_server_url=None, lr=0.0001,
                   optimizer_type='adam', owd=None, context_aware='NCA'):

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    all_metrics = []

    # Set up MLFlow server location, Otherwise location of ./mlruns
    mlflow.set_tracking_uri(mlflow_server_url)
    mlflow.set_experiment(experiment_name=mlflow_experiment_name)

    # Create a file to store the fold and repeat indices
    indices_save_path = os.path.join(output_dir, f"train_test_indices_{fe_taskname}.txt")
    with open(indices_save_path, 'w') as index_file:

        for repeat in range(n_repeats):
            # Start a nested MLFlow run for each repeat
            with mlflow.start_run(run_name=f"Repeat_{repeat + 1}_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}", nested=True):
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

                for fold, (train_index, test_index) in enumerate(skf.split(X, y)):
                    print(f"Fold {fold + 1}/{n_folds}")

                    # Log the indices of the training and test set for reproducibility
                    index_file.write(f"Repeat {repeat + 1}, Fold {fold + 1}\n")
                    index_file.write(f"Train indices: {train_index.tolist()}\n")
                    index_file.write(f"Test indices: {test_index.tolist()}\n\n")

                    # Log the train/test indices in MLFlow as well
                    mlflow.log_text(str(train_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_train_indices.txt")
                    mlflow.log_text(str(test_index.tolist()), f"Repeat_{repeat + 1}_Fold_{fold + 1}_test_indices.txt")

                    # Split the dataset into training and testing for this fold
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # Convert to PyTorch tensors
                    X_train, y_train = torch.tensor(X_train, dtype=torch.float32).to('cuda'), torch.tensor(y_train).to('cuda')
                    X_test, y_test = torch.tensor(X_test, dtype=torch.float32).to('cuda'), torch.tensor(y_test).to('cuda')

                    classifier = classifier.to('cuda')

                    if optimizer_type == "adam":
                        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=owd)
                    elif optimizer_type == "sgd":
                        optimizer = torch.optim.SGD(classifier.parameters, lr=lr, weight_decay=owd)

                    # Re-train the classifier for each fold
                    for epoch in range(epochs):
                        classifier.train()
                        epoch_loss = 0
                        for i in range(0, len(X_train), batch_size):
                            X_batch = X_train[i:i + batch_size]
                            y_batch = y_train[i:i + batch_size]

                            optimizer.zero_grad()
                            logits = classifier(X_batch)
                            loss = custom_categorical_cross_entropy(logits, y_batch, class_weights=class_weights)
                            loss.backward()
                            optimizer.step()

                            epoch_loss += loss.item()

                        # Log training metrics after each epoch
                        avg_epoch_loss = epoch_loss / len(X_train)
                        mlflow.log_metric("Train_Loss_"+str(fold), float(np.round(avg_epoch_loss, 4)), step=epoch)

                    # Evaluate on the test set
                    classifier.eval()
                    with torch.no_grad():
                        test_logits = classifier(X_test)
                        y_pred = torch.argmax(test_logits, dim=1).cpu().numpy()
                        y_true = y_test.cpu().numpy()

                        # Calculate metrics and confusion matrix
                        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y))
                        acc = accuracy_score(y_true, y_pred)
                        f1 = f1_score(y_true, y_pred, average='weighted')
                        precision = precision_score(y_true, y_pred, average='weighted')
                        recall = recall_score(y_true, y_pred, average='weighted')
                        # For binary classification, use the probability of the positive class (second column)
                        if len(np.unique(y_true)) == 2:
                            auc = roc_auc_score(y_true, F.softmax(test_logits, dim=1).cpu().numpy()[:, 1])
                        else:
                            # For multi-class, calculate AUC with `multi_class='ovr'`
                            auc = roc_auc_score(y_true, F.softmax(test_logits, dim=1).cpu().numpy(), multi_class='ovr')

                        # Accumulate confusion matrix
                        if cumulative_cm is None:
                            cumulative_cm = cm
                        else:
                            cumulative_cm += cm

                        # Log test metrics
                        mlflow.log_metric(f"Test_Accuracy_Repeat_{repeat+1}_Fold_{fold+1}", acc)
                        mlflow.log_metric(f"Test_F1_Repeat_{repeat+1}_Fold_{fold+1}", f1)
                        mlflow.log_metric(f"Test_Precision_Repeat_{repeat+1}_Fold_{fold+1}", precision)
                        mlflow.log_metric(f"Test_Recall_Repeat_{repeat+1}_Fold_{fold+1}", recall)
                        mlflow.log_metric(f"Test_AUC_Repeat_{repeat+1}_Fold_{fold+1}", auc)

                        fold_metrics.append({
                            'Repeat': repeat,
                            'Fold': fold,
                            'Accuracy': acc,
                            'F1': f1,
                            'Precision': precision,
                            'Recall': recall,
                            'AUC': auc
                        })

                # Log the average metrics for this repeat
                fold_metrics_df = pd.DataFrame(fold_metrics)
                avg_fold_metrics = fold_metrics_df.mean()
                mlflow.log_metric(f"Avg_Accuracy_Repeat_{repeat+1}", avg_fold_metrics['Accuracy'])
                mlflow.log_metric(f"Avg_F1_Repeat_{repeat+1}", avg_fold_metrics['F1'])
                mlflow.log_metric(f"Avg_Precision_Repeat_{repeat+1}", avg_fold_metrics['Precision'])
                mlflow.log_metric(f"Avg_Recall_Repeat_{repeat+1}", avg_fold_metrics['Recall'])
                mlflow.log_metric(f"Avg_AUC_Repeat_{repeat+1}", avg_fold_metrics['AUC'])

                # Accumulate metrics across repeats
                all_metrics.extend(fold_metrics)

                # Save cumulative confusion matrix after each repeat
                cm_image_path = os.path.join(output_dir, f"cumulative_confusion_matrix_repeat_{repeat+1}.png")
                plot_confusion_matrix(cumulative_cm, labels=np.unique(y), fe_taskname=fe_taskname, cm_image_path=cm_image_path)
                mlflow.log_artifact(cm_image_path, f"confusion_matrices/repeat_{repeat+1}")

                # End MLFlow run after each repeat
                mlflow.end_run()

            # Calculate overall average metrics across all repeats
            all_metrics_df = pd.DataFrame(all_metrics)
            overall_avg_metrics = all_metrics_df.mean()

            # Log overall average metrics to MLFlow
            mlflow.start_run(run_name=f"Overall_Avg_{context_aware}_{fe_taskname}_{optimizer_type}_{str(lr)}_{str(owd)}")

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
        metrics_df, cumulative_cm = monte_carlo_cv(
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
    parser.add_argument("--mlflow_experiment_name", default="[07092024] HPSearch Classifiers", type=str,
                        help='Name for experiment in MLFlow') #[Final] Classifier on Final CBDC 06_09_2024
    parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

    # General params
    parser.add_argument('--output_dir', type=str, default="./results", help='Path to save results')
    parser.add_argument('--context_aware', default="CA", type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
    parser.add_argument('--n_folds', default=2, type=int, help='Number of folds for Monte Carlo CV')
    parser.add_argument('--n_repeats', default=2, type=int, help='Number of Monte Carlo repeats')
    parser.add_argument('--gt_path', default="../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx", type=str, help='Path to ground truth file')
    parser.add_argument('--graphs_dir', default="../data/CLARIFY/results_graphs_november_23", type=str, help='Directory where graphs are stored')
    parser.add_argument('--knn', default=19, type=int, help='KNN used to store graphs')
    parser.add_argument('--pretrained_model_path', type=str, help='Path to pretrained model')
    parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate of the classifier')
    parser.add_argument('--optimizer_type', default="adam", type=str, help='Optimizer type')
    parser.add_argument('--owd', default=None, type=float, help='Optimizer weight decay')

    # Parse the fixed arguments
    args = parser.parse_args()

    # Lists of hyperparameters to loop over
    lrs = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    optimizers = ["adam", "sgd"]
    owds = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
    epochs = [200, 500]
    batch_sizes = [64, 128, 256]
    context_awareness = ["CA", "NCA"]


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
