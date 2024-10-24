from MIL_utils import *
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import os
import re
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
from sklearn.metrics import make_scorer, f1_score
from collections import Counter
#from sklearn.metrics import roc_auc_score
import seaborn as sns
#from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.utils._testing import ignore_warnings
#from sklearn.exceptions import ConvergenceWarning
#@ignore_warnings(category=ConvergenceWarning)
import mlflow

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import cross_val_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight

class FlexibleClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes, activations):
        super(FlexibleClassifier, self).__init__()

        assert len(hidden_sizes) == len(activations), "Number of hidden sizes must match the number of activation functions."

        self.layers = nn.ModuleList()
        prev_size = input_size

        for size, activation_fn in zip(hidden_sizes, activations):
            self.layers.append(nn.Linear(prev_size, size))
            if activation_fn is not None:
                self.layers.append(activation_fn)
            prev_size = size

        # Output layer
        self.output_layer = nn.Linear(prev_size, num_classes)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        output = self.output_layer(x)
        return output


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
    #y_pred = torch.unsqueeze(y_pred, 0)

    # Choose the loss function based on the specified loss_function parameter.
    if loss_function == "cross_entropy":
        loss_not_balanced = torch.nn.CrossEntropyLoss()
    elif loss_function == "kll":
        loss_not_balanced = torch.nn.KLDivLoss()
    elif loss_function == "mse":
        loss_not_balanced = torch.nn.MSELoss()

    # Compute the unweighted loss using the chosen loss function.

    loss = loss_not_balanced(y_pred, y_true)
    y_pred_hat = torch.argmax(y_pred, dim=1)

    try:
        # If class weights are specified, compute the weight for the actual class and apply it to the loss.
        if class_weights is not None:
            weight_actual_class = class_weights[y_true]
            loss = loss * weight_actual_class
    except RuntimeError:
        return loss.mean()
        print("hola")

    # Return the computed loss.
    return loss.mean()

def plot_confusion_matrices(confusion_matrices, label_mappings=None):
    """
    Plot confusion matrices for each model in the dictionary.

    Parameters:
    - confusion_matrices (dict): A dictionary where keys are model names and values are confusion matrices.
    - label_mappings (dict, optional): A dictionary containing label mappings for each classification task.
    """
    for model_name, confusion_matrix in confusion_matrices.items():
        # Create a DataFrame from the confusion matrix with class labels
        confusion_matrix_df = pd.DataFrame(confusion_matrix, columns=label_mappings, index=label_mappings)

        # Set up the matplotlib figure
        plt.figure(figsize=(8, 6))

        # Plot the confusion matrix with integer values
        sns.heatmap(confusion_matrix_df, annot=True, cmap="Blues", fmt="d", xticklabels=True, yticklabels=True)


        # Add labels based on label_mappings if provided
        # if label_mappings:
        #     plt.xticks(np.arange(len(label_mappings)), label_mappings, rotation=45)
        #     plt.yticks(np.arange(len(label_mappings)), label_mappings, rotation=0)

        plt.title(f'Confusion Matrix - {model_name}')
        #plt.xlabel('Predicted Label')
        #plt.ylabel('True Label')

        # Display the plot
        plt.show()

def load_and_preprocess_clarify_data(seed, data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=seed, stratify=target)
    return X_train, X_test, y_train, y_test

# Main function
def main(args):

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Read ground truth
    #gt_df = pd.read_excel("../data/BCNB/ground_truth/patient-clinical-data.xlsx")
    #gt_df = pd.read_excel("../data/CLARIFY/ground_truth/final_clinical_info_CLARIFY_DB.xlsx")
    gt_df = pd.read_excel("../data/CLARIFY/ground_truth/CBDC_4_may2024_gt_extended.xlsx")


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
    #graphs_dirs.reverse()


    for graph_dirname in graphs_dirs:

        # Depending on feature extractor select task name and labels
        fe_taskname = graph_dirname.split("_BB")[0].split("graphs_PM_")[1]
        task_labels_mapping = tasks_labels_mappings[fe_taskname]

        # Derive feature extractor used for extracting features to build the graphs
        # extracted_features_savepath = os.path.join(args.feats_savedir, )
        try:
            if args.context_aware == "NCA":
                chosen_model = [fe_name for fe_name in feature_extractors if fe_taskname in fe_name][0]
                chosen_model_path = os.path.join(args.feature_extractors_dir, chosen_model)

            elif args.context_aware == "CA":
                chosen_model = [fe_name for fe_name in feature_extractors if fe_taskname in fe_name][0]
                args.knn = chosen_model.split("KNN_")[1].split("_")[0]
                chosen_model_path = os.path.join(args.pretrained_gcn_models_dir, chosen_model)
        except IndexError:
            continue

        print("Chosen model: ", chosen_model)
        # Load pretrained model
        model = torch.load(chosen_model_path).to('cuda')

        #  Iterate over graphs
        graphs_knn_dir = os.path.join(args.graphs_dir, graph_dirname, "graphs_k_" + str(args.knn))
        print("Chosen graphs: ", graphs_knn_dir)
        graphs_files = os.listdir(graphs_knn_dir)


        # Filter out graphs that are "Excluded"
        # Extract patient IDs from filenames
        def extract_patient_id(filename):
            match = re.search(r'(SUS\d+)', filename)
            return match.group(1) if match else None

        # Create a DataFrame from the graph files list
        graph_files_df = pd.DataFrame({
            'filename': graphs_files,
            'SUS_number': [extract_patient_id(filename) for filename in graphs_files]
        })

        # Merge the graph files DataFrame with the ground truth DataFrame
        merged_df = pd.merge(graph_files_df, gt_df, on='SUS_number', how='inner')

        # Filter out rows where Molsub_surr_3clf is "Excluded"
        filtered_df = merged_df[merged_df['Molsub_surr_7clf'] != 'Excluded']

        # Extract the filenames from the filtered DataFrame
        filtered_files = filtered_df['filename'].tolist()

        graphs_files = filtered_files

        # Initialize lists to store features, labels, and task labels
        all_features = []
        all_labels = []
        all_task_labels = []

        for graph_name in tqdm(graphs_files):

            #for task, labels_mappings in tasks_labels_mappings.items():

            max_n_cases = 5
            counter = 0

            #file_id = graph_name.split("_")[0]
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
            #filtered_imgs_paths = [path for path in selected_images_paths if file_id in path.split("/")[-2]]
            input_shape = (3, 256, 256)

            #case_features = []
            # Read all images paths
            with torch.no_grad():

                if args.context_aware == "NCA":
                    # Aggregate using pretrained attention
                    case_aggr_feature_vector = model.milAggregation(graph_features)

                elif args.context_aware == "CA":
                    _, _, _, case_aggr_feature_vector = model(graph)

            # Find matching label for graph
            #id_label = gt_df[gt_df["Patient ID"] == int(file_id)]["Molecular subtype"].values[0]
            id_label = gt_df[gt_df["SUS_number"] == file_id]["Molsub_surr_4clf"].values[0]

            # Encode task label to numeric value
            encoded_task_label = task_labels_mapping.get(id_label, 0)  # Use 0 as a default value if the label is not found

            # Append features, labels, and task labels
            all_features.append(case_aggr_feature_vector.cpu().detach().numpy())
            all_labels.append(encoded_task_label)
            all_task_labels.append(encoded_task_label)

            counter += 1

        # Concatenate features for t-SNE
        tsne_features = np.stack(all_features)
        all_labels = np.array(all_labels)

        # Load and preprocess data
        #X_train_all, X_test_all, Y_train_all, Y_test_all = load_and_preprocess_clarify_data(seed=args.seed, data=tsne_features, target=all_labels)

        # train linear classifier
        nClasses = len(np.unique(all_labels))
        #classifier = torch.nn.Linear(X_train_all.shape[1], nClasses).to('cuda')
        #optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

        # Assuming X_train, Y_train, X_test, Y_test are PyTorch tensors
        # Adjust batch_size based on your preferences
        batch_size = args.batch_size
        num_epochs = args.epochs

        # Create an empty DataFrame to store metrics
        columns = ['Fold', 'Accuracy', 'AUC', 'Weighted F1', 'Precision', 'Recall']
        metrics_df = pd.DataFrame(columns=columns)



        # Get MLFlow arguments
        mlflow_experiment_name = args.mlflow_experiment_name
        mlflow_server_url = args.mlflow_server_url

        # Set up MLFlow server location, Otherwise location of ./mlruns
        mlflow.set_tracking_uri(mlflow_server_url)

        # Create MLFlow experiment
        mlflow.set_experiment(experiment_name=mlflow_experiment_name)

        # Start training

        print(f"Training for final CBDC split.")

        mlflow_run_name = str(fe_taskname) + "_LR_" + str(args.lr).replace(".", "") + "_Optim_" + str(
            args.optimizer_type) + "_OWD_" + str(args.owd) + "_FinalSplit"

        # Start MLFlow run
        mlflow.start_run(run_name=mlflow_run_name)  # If not provided, run_name will be randomly assigned

        # Log MLFlow Parameters
        for key, value in vars(args).items():
            mlflow.log_param(key, value)
        mlflow.log_param("Taskname", fe_taskname)

        #TODO: adapt for final split

        # Split data into training-validation-test sets
        X_train_all, X_test, Y_train_all, Y_test = train_test_split(tsne_features, all_labels, test_size=args.test_size,
                                                                    random_state=args.seed, stratify=all_labels)

        # Split data into training and validation sets for the current fold
        X_train, Y_train = X_train_all, Y_train_all
        X_test, Y_test = X_test, Y_test

        # Compute class weights for balancing loss
        class_weights = compute_class_weight('balanced', classes=np.unique(Y_train),
                                             y=Y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to('cuda')

        # Move data to GPU if available
        X_train, Y_train = torch.tensor(X_train).to('cuda'), torch.tensor(Y_train).to('cuda')
        X_test, Y_test = torch.tensor(X_test).to('cuda'), torch.tensor(Y_test)

        # Initialize model and optimizer for each fold
        #classifier = torch.nn.Linear(X_train.shape[1], nClasses).to('cuda')
        classifier = FlexibleClassifier(input_size=X_train.shape[1], num_classes=nClasses, hidden_sizes=args.hidden_size, activations=args.activations).to('cuda')

        # Define optimizer
        if args.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(classifier.parameters(), lr=args.lr, weight_decay=args.owd)
        elif args.optimizer_type == "adam":
            optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.owd)

        #optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.owd)

        # Track the best model weights
        best_val_f1_score = 0.0
        best_model_weights = None

        # Training loop
        for epoch in range(num_epochs):
            classifier.train()
            accumulated_loss = 0.0
            refs_train = []
            preds_train = []
            preds_prob_train = []


            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i:i + batch_size]
                Y_batch = Y_train[i:i + batch_size]

                train_logits = classifier(X_batch)
                train_Y_hat = torch.argmax(train_logits, dim=1)
                train_Y_prob = F.softmax(train_logits, dim=1)

                preds_train.append(train_Y_hat.detach().cpu().numpy())
                preds_prob_train.append(train_Y_prob.detach().cpu().numpy())
                refs_train.append(Y_batch.detach().cpu().numpy())

                L_train = custom_categorical_cross_entropy(y_pred=train_logits.squeeze(),
                                                     y_true=Y_batch,
                                                     class_weights=class_weights,
                                                     loss_function="cross_entropy")
                accumulated_loss += L_train.item()

                optimizer.zero_grad()
                L_train.backward()
                optimizer.step()


            # Epoch end
            refs_train = np.concatenate(np.array(refs_train))
            preds_train = np.concatenate(np.array(preds_train))
            preds_prob_train = np.concatenate(np.array(preds_prob_train))


            train_accuracy = accuracy_score(refs_train, preds_train)
            if nClasses > 2:
                train_auc = roc_auc_score(refs_train, preds_prob_train,
                                          multi_class='ovr')
            else:
                train_y_score_positive_class = preds_prob_train[:, 1]
                train_auc = roc_auc_score(refs_train, train_y_score_positive_class)

            train_weighted_f1_score = f1_score(refs_train, preds_train,
                                               average='weighted')
            train_precision = precision_score(refs_train, preds_train,
                                              average='weighted')
            train_recall = recall_score(refs_train, preds_train, average='weighted')

            # Calculate average loss per epoch
            L_epoch = accumulated_loss / (len(X_train) / batch_size)

            # Log to MLFlow
            mlflow.log_metric("train_auc", train_auc, step=epoch)
            mlflow.log_metric("train_weighted_f1_score", train_weighted_f1_score, step=epoch)
            mlflow.log_metric("train_precision", train_precision, step=epoch)
            mlflow.log_metric("train_recall", train_recall, step=epoch)
            mlflow.log_metric("L_train_epoch", L_epoch, step=epoch)

            # Save weights after epoch
            best_model_weights = classifier.state_dict()

        # Load the best model weights for the current fold
        classifier.load_state_dict(best_model_weights)

        # Now you can use the fixed independent test set (X_test, Y_test) for final evaluation
        with torch.no_grad():
            classifier.eval()

            test_logits = classifier(torch.tensor(X_test).to('cuda'))
            test_predictions = torch.argmax(test_logits, dim=1)
            test_prob = F.softmax(test_logits, dim=1)

            test_accuracy = accuracy_score(Y_test, test_predictions.cpu().detach().numpy())
            if nClasses > 2:
                test_auc = roc_auc_score(Y_test, test_prob.cpu().detach().numpy(), multi_class='ovr')
            else:
                y_score_positive_class = test_prob.cpu().detach().numpy()[:, 1]
                test_auc = roc_auc_score(Y_test, y_score_positive_class)

            test_weighted_f1_score = f1_score(Y_test, test_predictions.cpu().numpy(), average='weighted')
            test_precision = precision_score(Y_test, test_predictions.cpu().numpy(), average='weighted')
            test_recall = recall_score(Y_test, test_predictions.cpu().numpy(), average='weighted')

            mlflow.log_metric("test_auc", test_auc, step=epoch)
            mlflow.log_metric("test_weighted_f1_score", test_weighted_f1_score, step=epoch)
            mlflow.log_metric("test_precision", test_precision, step=epoch)
            mlflow.log_metric("test_recall", test_recall, step=epoch)



            print(
                f"Final Test Accuracy: {test_accuracy:.4f}, AUC: {test_auc:.4f}, F1: {test_weighted_f1_score:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")

            # Append metrics to the DataFrame
            metrics_df = metrics_df.append({
                'Fold': 0,
                'Accuracy': test_accuracy,
                'AUC': test_auc,
                'Weighted F1': test_weighted_f1_score,
                'Precision': test_precision,
                'Recall': test_recall
            }, ignore_index=True)

            cm_test = confusion_matrix(Y_test, test_predictions.cpu().numpy())
            print(f"Confusion Matrix - Final Test Set:")
            print(cm_test)

            # Create output directory
            os.makedirs(os.path.join(args.output_dir, mlflow_experiment_name, mlflow_run_name), exist_ok=True)

            cf_savepath = os.path.join(args.output_dir, mlflow_experiment_name, mlflow_run_name, f"Confusion_Matrix_Final_Test.png")
            # Display the confusion matrix using seaborn heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm_test, annot=True, fmt="d", cmap="Blues", xticklabels=task_labels_mapping,
                        yticklabels=task_labels_mapping)
            # plt.xlabel("Predicted Label")
            # plt.ylabel("True Label")
            plt.title(f"Test Set")
            plt.savefig(cf_savepath, bbox_inches='tight')
            mlflow.log_artifact(cf_savepath, f"test_cf_final_test_set")
            #plt.show()

            mlflow.end_run()

        # Calculate mean and standard deviation of metrics
        print(metrics_df)
        mean_metrics = metrics_df.mean()
        std_metrics = metrics_df.std()

        print("\nMean Metrics:")
        print(mean_metrics)

        print("\nStandard Deviation Metrics:")
        print(std_metrics)
    print("hola")

# Run the main function
if __name__ == "__main__":

    optimizers = ["adam"] #, "sgd"]
    lrs = [0.001, 0.0001, 0.00001] #[0.001]
    owds = [ 0.0001, 0.00001]# [0, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    batch_sizes = [64, 128]
    hidden_sizes = [[128], [256]]# [[256], [128], [64]]
    activations = [[nn.ReLU()], [nn.Tanh()]]
    cas = ["CA", "NCA"]

    for optimizer in optimizers:
        for lr in lrs:
            for owd in owds:
                for batch_size in batch_sizes:
                    for hidden_size in hidden_sizes:
                        for activation in activations:
                            for ca in cas:
                                ##########################
                                # CREATE ARGUMENT PARSER #
                                ##########################
                                parser = argparse.ArgumentParser()

                                # MLFlow configuration
                                parser.add_argument("--mlflow_experiment_name", default="[Dev] Classifier on Final CBDC 28_07 ", type=str,
                                                    help='Name for experiment in MLFlow')
                                parser.add_argument('--mlflow_server_url', type=str, default="http://158.42.170.104:8002", help='URL of MLFlow DB')

                                # General params
                                parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
                                parser.add_argument('--path_to_bcnb_dataset', type=str, default="C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB", help="path where h5 files are located")
                                parser.add_argument('--path_to_feature_extractors_folder', type=str, default="/Molecular_Subtype_Prediction/data/feature_extractors", help="path where feature extractors are located")
                                parser.add_argument('--seed', type=int, default=47, help="seed for reproducibility")
                                parser.add_argument('--feats_savedir', type=str, default="../data/extracted_features/NCA_WSI_feats", help="path to save features folder")
                                parser.add_argument('--output_dir', type=str, default="../data/outputs", help="path to savefigs")

                                # Feature extractor params
                                #parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str, help='Classification task') #
                                parser.add_argument("--context_aware", default=ca, type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
                                parser.add_argument("--graphs_dir", default="../data/CLARIFY/results_graphs_november_23", type=str, help='Directory where graphs are stored.') # "C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB/results_graphs_november_23"
                                #parser.add_argument("--knn", default=25, type=int, help='KNN used to store graphs.')
                                parser.add_argument("--epochs", default=100, type=int, help='KNN used to store graphs.')
                                parser.add_argument("--batch_size", default=batch_size, type=int, help='KNN used to store graphs.')

                                parser.add_argument("--n_kfolds", default=5, type=float, help='number of K folds for the Cross Validatio')
                                parser.add_argument("--test_size", default=0.3, type=float, help='Percentage of the full test_size')
                                parser.add_argument("--lr", default=lr, type=float, help='KNN used to store graphs.')
                                parser.add_argument("--owd", default=owd, type=float, help='KNN used to store graphs.')
                                parser.add_argument("--knn", default=19, type=int, help='KNN used to store graphs.')
                                parser.add_argument("--optimizer_type", default=optimizer, type=str, help='adam or sgd')

                                parser.add_argument("--hidden_size", default=hidden_size, type=list, help='hidden dimension of classifier')
                                parser.add_argument("--activations", default=activation, type=list, help='activation introduces non-linearity')

                                args = parser.parse_args()
                                main(args)