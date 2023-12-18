from MIL_utils import *
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
import os
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neighbors
import sklearn.svm
import sklearn.ensemble
import sklearn.neural_network
import sklearn.model_selection
from sklearn.metrics import make_scorer, f1_score

import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

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

# ML models class
class supervised_models:
    '''Set of supervised learning models class'''

    def __init__(self, n_seed=0):
        '''Initializing necessary variables'''
        self.num_cores = os.cpu_count()
        self.seed = n_seed

    # models
    def models(self, cv=5, model='lr'):
        '''Classifier models of supervised learning with with GridSearchCV optimization.
        INPUTS:
        X_train: from train-test split
        y_train: from train-test split
        cv: Number of folds in the cross-validation
        model: can be 'lr', 'svm', 'rf', 'xgb', 'nb', 'knn', 'sgd', 'mlp'.
        OUTPUT: trained model'''

        # Define models and search spaces
        if model == 'lr':
            clf = sklearn.linear_model.LogisticRegression(multi_class='auto')
            param_grid = {'penalty': ['none', 'l1', 'l2', 'elasticnet'],
                          'class_weight': ['balanced', None],
                          'solver': ['saga'],
                          'max_iter': [1000],
                          'l1_ratio': [0, 0.25, 0.5, 0.75, 1],
                          'random_state': [self.seed]}

        elif model == 'nb':
            clf = sklearn.naive_bayes.GaussianNB()
            param_grid = {'var_smoothing': [1e-8, 1e-9, 1e-10]}

        elif model == 'knn':
            clf = sklearn.neighbors.KNeighborsClassifier()
            param_grid = {'n_neighbors': [3, 5, 7, 9],
                          'weights': ['uniform', 'distance'],
                          'p': [1, 2, 3]}

        elif model == 'svm':
            clf = sklearn.svm.SVC()
            param_grid = {'C': [0.1, 1, 10, 100, 1000],
                          'gamma': [1, 0.1, 0.01, 0.0001],
                          'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                          'class_weight': ['balanced', None],
                          'random_state': [self.seed]}

        elif model == 'rf':
            clf = sklearn.ensemble.RandomForestClassifier()
            param_grid = {'n_estimators': [50, 100, 200, 500],
                          'max_depth': [2, 4, 6, 8, None],
                          'class_weight': ['balanced', None],
                          'random_state': [self.seed]}

        elif model == 'xgb':
            clf = sklearn.ensemble.GradientBoostingClassifier()
            param_grid = {'learning_rate': [0.001, 0.01, 0.1, 1, 10],
                          'n_estimators': [50, 100, 200, 500],
                          'min_samples_split': [2, 4],
                          'max_depth': [2, 3, 4],
                          'random_state': [self.seed]}

        elif model == 'sgd':
            clf = sklearn.linear_model.SGDClassifier()
            param_grid = {'loss': ['hinge', 'log', 'modified_huber'],
                          'penalty': ['l1', 'l2', 'elasticnet'],
                          'alpha': [0.0001, 0.001, 0.01, 0.1],
                          'max_iter': [1000],
                          'random_state': [self.seed]}

        elif model == 'mlp':
            clf = sklearn.neural_network.MLPClassifier()
            param_grid = {'hidden_layer_sizes': [(50, 50), (100, 100)],
                          'activation': ['relu', 'tanh'],
                          'alpha': [0.0001, 0.001, 0.01],
                          'max_iter': [200, 500],
                          'random_state': [self.seed]}

        # Define a CV grid-search
        f1_scorer = make_scorer(f1_score, average='weighted')  # Use weighted F1-score for multiclass

        # Define a CV grid-search
        clf_grid = sklearn.model_selection.GridSearchCV(clf,
                                                   param_grid,
                                                   scoring={'f1_weighted': f1_scorer},
                                                   cv=cv,
                                                   n_jobs=self.num_cores - 1,
                                                   verbose=1,
                                                   refit='f1_weighted')

        # Return model specifications
        return (clf_grid)

# Function to load and preprocess data (example using Iris dataset)
def load_and_preprocess_data(seed):
    iris = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test

def load_and_preprocess_clarify_data(seed, data, target):
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=seed)
    return X_train, X_test, y_train, y_test

def train_ml_classifiers(X_train, X_test, y_train, y_test):
    # Initialize the supervised_models class
    model_handler = supervised_models(n_seed=47)

    # Try Logistic Regression
    lr_model = model_handler.models(cv=5, model='lr')
    lr_model.fit(X_train, y_train)
    lr_predictions = lr_model.predict(X_test)
    print("Logistic Regression ")
    print(classification_report(y_test, lr_predictions))

    print("Logistic Regression Best Model:")
    lr_best_model = lr_model.best_estimator_
    lr_best_model.fit(X_train, y_train)
    lr_best_predictions = lr_model.predict(X_test)
    print("Logistic Regression Best CV:")
    print(classification_report(y_test, lr_best_predictions))
    print("################################################")

    # Try Naive Bayes
    nb_model = model_handler.models(cv=5, model='nb')
    nb_model.fit(X_train, y_train)
    nb_predictions = nb_model.predict(X_test)
    print("Naive Bayes ")
    print(classification_report(y_test, nb_predictions))

    print("NB Best Model:")
    nb_best_model = nb_model.best_estimator_
    nb_best_model.fit(X_train, y_train)
    nb_best_predictions = nb_model.predict(X_test)
    print("NB Best CV:")
    print(classification_report(y_test, nb_best_predictions))
    print("################################################")

    # Try KNN
    knn_model = model_handler.models(cv=5, model='knn')
    knn_model.fit(X_train, y_train)
    knn_predictions = knn_model.predict(X_test)
    print("KNeighbors ")
    print(classification_report(y_test, knn_predictions))

    print("KNN Best Model:")
    knn_best_model = knn_model.best_estimator_
    knn_best_model.fit(X_train, y_train)
    knn_best_predictions = knn_model.predict(X_test)
    print("KNN Best CV:")
    print(classification_report(y_test, knn_best_predictions))
    print("################################################")

    # Try Random Forest
    rf_model = model_handler.models(cv=5, model='rf')
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    print("Random Forest ")
    print(classification_report(y_test, rf_predictions))

    print("RF Best Model:")
    rf_best_model = rf_model.best_estimator_
    rf_best_model.fit(X_train, y_train)
    rf_best_predictions = rf_model.predict(X_test)
    print("RF Best CV:")
    print(classification_report(y_test, rf_best_predictions))
    print("################################################")

    # Try XGBoost
    xgb_model = model_handler.models(cv=5, model='xgb')
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    print("XGBoost ")
    print(classification_report(y_test, xgb_predictions))

    print("XGBoost Best Model:")
    xgb_best_model = xgb_model.best_estimator_
    xgb_best_model.fit(X_train, y_train)
    xgb_best_predictions = xgb_model.predict(X_test)
    print("XGBoost Best CV:")
    print(classification_report(y_test, xgb_best_predictions))
    print("################################################")

    # Try SGD
    sgd_model = model_handler.models(cv=5, model='sgd')
    sgd_model.fit(X_train, y_train)
    sgd_predictions = sgd_model.predict(X_test)
    print("SGD ")
    print(classification_report(y_test, sgd_predictions))

    print("SGD Best Model:")
    sgd_best_model = sgd_model.best_estimator_
    sgd_best_model.fit(X_train, y_train)
    sgd_best_predictions = sgd_model.predict(X_test)
    print("SGD Best CV:")
    print(classification_report(y_test, sgd_best_predictions))
    print("################################################")

    # Try Multi-layer Perceptron
    mlp_model = model_handler.models(cv=5, model='mlp')
    mlp_model.fit(X_train, y_train)
    mlp_predictions = mlp_model.predict(X_test)
    print("Multi-layer Perceptron:")
    print(classification_report(y_test, mlp_predictions))

    print("MLP Best Model:")
    mlp_best_model = mlp_model.best_estimator_
    mlp_best_model.fit(X_train, y_train)
    mlp_best_predictions = mlp_model.predict(X_test)
    print("MLP Best CV:")
    print(classification_report(y_test, mlp_best_predictions))
    print("################################################")


    # Try Support Vector Machine
    svm_model = model_handler.models(cv=5, model='svm')
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    print("Support Vector Machine:")
    print(classification_report(y_test, svm_predictions))

    print("SVM Best Model:")
    svm_best_model = svm_model.best_estimator_
    svm_best_model.fit(X_train, y_train)
    svm_best_predictions = svm_model.predict(X_test)
    print("SVM Best CV:")
    print(classification_report(y_test, svm_best_predictions))
    print("################################################")

    print("hola")

def train_ml_classifiers_df(X_train, X_test, y_train, y_test):
    results = []
    full_classification_reports = {}
    confusion_matrices = {}

    def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
        print("Training with: ", model_name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(y_test, predictions, output_dict=True)
        full_classification_reports[model_name] = report

        # Compute confusion matrix
        cm = confusion_matrix(y_test, predictions)
        confusion_matrices[model_name] = cm

        return {
            'Model': model_name,
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'Support': report['weighted avg']['support']
        }

    # Initialize the supervised_models class
    model_handler = supervised_models(n_seed=47)

    # # Logistic Regression
    # lr_model = model_handler.models(cv=5, model='lr')
    # lr_results = train_and_evaluate('Logistic Regression', lr_model, X_train, y_train, X_test, y_test)
    # results.append(lr_results)

    # Naive Bayes
    nb_model = model_handler.models(cv=5, model='nb')
    nb_results = train_and_evaluate('Naive Bayes', nb_model, X_train, y_train, X_test, y_test)
    results.append(nb_results)

    # K-Nearest Neighbors (KNN)
    knn_model = model_handler.models(cv=5, model='knn')
    knn_results = train_and_evaluate('KNN', knn_model, X_train, y_train, X_test, y_test)
    results.append(knn_results)

    # # Random Forest
    # rf_model = model_handler.models(cv=5, model='rf')
    # rf_results = train_and_evaluate('Random Forest', rf_model, X_train, y_train, X_test, y_test)
    # results.append(rf_results)

    # # XGBoost
    # xgb_model = model_handler.models(cv=5, model='xgb')
    # xgb_results = train_and_evaluate('XGBoost', xgb_model, X_train, y_train, X_test, y_test)
    # results.append(xgb_results)

    # # Stochastic Gradient Descent (SGD)
    # sgd_model = model_handler.models(cv=5, model='sgd')
    # sgd_results = train_and_evaluate('SGD', sgd_model, X_train, y_train, X_test, y_test)
    # results.append(sgd_results)

    # # Multi-layer Perceptron (MLP)
    # mlp_model = model_handler.models(cv=5, model='mlp')
    # mlp_results = train_and_evaluate('Multi-layer Perceptron', mlp_model, X_train, y_train, X_test, y_test)
    # results.append(mlp_results)
    #
    # # Support Vector Machine (SVM)
    # svm_model = model_handler.models(cv=5, model='svm')
    # svm_results = train_and_evaluate('Support Vector Machine', svm_model, X_train, y_train, X_test, y_test)
    # results.append(svm_results)

    # Create DataFrame
    results_df = pd.DataFrame(results)

    return results_df, full_classification_reports, confusion_matrices

# Main function
def main(args):

    # Load CLARIFY Data

    # Read ground truth
    #gt_df = pd.read_excel("../data/BCNB/ground_truth/patient-clinical-data.xlsx")
    gt_df = pd.read_excel("../data/CLARIFY/ground_truth/final_clinical_info_CLARIFY_DB.xlsx")

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
            id_label = gt_df[gt_df["SUS_number"] == file_id]["MolSubtype_surr"].values[0]

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
        X_train, X_test, y_train, y_test = load_and_preprocess_clarify_data(seed=47, data=tsne_features, target=all_labels)

        # Example usage:
        df_results, classification_reports, confusion_matrices = train_ml_classifiers_df(X_train, X_test, y_train, y_test)
        print(df_results)
        print(classification_reports)
        print(confusion_matrices)
        plot_confusion_matrices(confusion_matrices, label_mappings=tasks_labels_mappings[fe_taskname])

# Run the main function
if __name__ == "__main__":

    ##########################
    # CREATE ARGUMENT PARSER #
    ##########################
    parser = argparse.ArgumentParser()

    # General params
    parser.add_argument('--where_exec', type=str, default="local", help="slurm_dgx, slurm_nas, dgx_gpu or local")
    parser.add_argument('--path_to_bcnb_dataset', type=str, default="C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB", help="path where h5 files are located")
    parser.add_argument('--path_to_feature_extractors_folder', type=str, default="/Molecular_Subtype_Prediction/data/feature_extractors", help="path where feature extractors are located")
    #parser.add_argument('--tsne_savedir', type=str, default="/output_tsnes", help="path to save graphs folder")
    parser.add_argument('--feats_savedir', type=str, default="../data/extracted_features/NCA_WSI_feats", help="path to save features folder")

    # Feature extractor params
    #parser.add_argument("--pred_mode", default="OTHERvsTNBC", type=str, help='Classification task') #
    parser.add_argument("--context_aware", default="NCA", type=str, help='Context-aware (CA) or Non-Context-Aware (NCA)')
    parser.add_argument("--graphs_dir", default="../data/CLARIFY/results_graphs_november_23", type=str, help='Directory where graphs are stored.') # "C:/Users/clferma1/Documents/Investigacion_GIT/Molecular_Subtype_Prediction/data/BCNB/results_graphs_november_23"
    parser.add_argument("--knn", default=25, type=int, help='KNN used to store graphs.')

    args = parser.parse_args()
    main(args)

    #main()