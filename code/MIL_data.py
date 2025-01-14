from MIL_utils import *
from torch.utils.data import Subset, DataLoader


class MILDataset_offline_graphs(object):

    def __init__(self, args, graph_dirname, gt_df, task_labels_mapping, graphs_on_ram=False):
        self.args = args
        self.graphs_dir = args.graphs_dir
        self.graph_dirname = graph_dirname
        self.gt_df = gt_df
        self.task_labels_mapping = task_labels_mapping
        self.graphs_on_ram = graphs_on_ram

        # Load graphs for the task
        graphs_knn_dir = os.path.join(self.graphs_dir, graph_dirname, "graphs_k_" + str(args.knn))
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
        filtered_df = merged_df[merged_df['Molsub_surr_7clf'] != 'Excluded'][:537]

        # Preallocate Graphs on RAM if the flag is set
        self.graph_paths = []
        self.labels = []
        self.graphs = []

        if self.graphs_on_ram:
            for graph_name in tqdm(filtered_df['filename'].tolist()):
                file_id = graph_name.split("-")[0].split("HE")[0].split("_")[0].split("a")[0]
                file_path = os.path.join(graphs_knn_dir, graph_name)

                # Load the graph
                graph = torch.load(file_path).to('cuda')

                # Get the corresponding label for this case
                id_label = gt_df[gt_df["SUS_number"] == file_id]["Molsub_surr_4clf"].values[0]
                encoded_task_label = task_labels_mapping.get(id_label, 0)

                # Store graph and label
                self.graphs.append(graph)
                self.labels.append(encoded_task_label)
                self.graph_paths.append(file_path)

            # Ensure the data is collected correctly
            assert len(self.graphs) == len(self.labels), "Data size mismatch!"

        else:
            self.filtered_df = filtered_df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.graphs) if self.graphs_on_ram else len(self.filtered_df)

    def __getitem__(self, index):
        'Generates one sample of data'
        if self.graphs_on_ram:
            graph = self.graphs[index]
            label = self.labels[index]
        else:
            # Lazy loading of the graph
            graph_name = self.filtered_df.iloc[index]['filename']
            file_id = graph_name.split("-")[0].split("HE")[0].split("_")[0].split("a")[0]
            file_path = os.path.join(self.args.graphs_dir, self.graph_dirname, "graphs_k_" + str(self.args.knn), graph_name)

            # Load the graph
            graph = torch.load(file_path).to('cuda')

            # Get the corresponding label
            id_label = self.gt_df[self.gt_df["SUS_number"] == file_id]["Molsub_surr_4clf"].values[0]
            label = self.task_labels_mapping.get(id_label, 0)

        return graph, label

class MILDataGenerator_offline_graphs_balanced_all_classes(object):
    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.balanced_class_idx)  # or re-draw it
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, graphs_on_ram,
                 batch_size=1, shuffle=False, max_instances=100):
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset  # could be a Subset or the original dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        self.graphs_on_ram = graphs_on_ram

        # Distinguish Subset vs. full dataset
        if isinstance(dataset, Subset):
            self.parent_dataset = dataset.dataset
            self.subset_indices = dataset.indices
        else:
            self.parent_dataset = dataset
            self.subset_indices = range(len(self.parent_dataset))

        self.d_len = len(self.subset_indices)

        # Prepare labels for just those subset indices
        self.d_classes = [self.parent_dataset.labels[i] for i in self.subset_indices]

        # Adjust label filtering based on pred_mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            # no change
            pass
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = [
                "Luminal" if x in ["Luminal A", "Luminal B"] else x
                for x in self.d_classes
            ]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = [
                "Other" if x in ["Luminal A", "Luminal B", "HER2(+)"] else x
                for x in self.d_classes
            ]

        # Create a dictionary from local indices -> label
        # i.e., in the subset
        self.local_idx_to_label = dict(zip(range(self.d_len), self.d_classes))

        # Compute class weights
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        self.weights = compute_class_weight(
            class_weight="balanced",
            classes=self.ordered_clases,
            y=self.d_classes
        )
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # We create a balanced_class_idx (long list) to indicate which class to pick next
        self.balanced_class_idx = torch.multinomial(
            input=self.weights_t,
            num_samples=self.d_len,  # or bigger if you want more draws
            replacement=True
        )

        # For ensuring we don't pick the same graph_idx repeatedly for the same class
        self.last_graph_idxs = {}

        # Initialize iteration
        self._reset()

    def __next__(self):
        # If we've exhausted the dataset length, stop
        if self._idx >= self.d_len:
            self._reset()
            raise StopIteration()

        # We'll collect `self.batch_size` items
        batch_graphs = []
        batch_labels = []

        for i in range(self.batch_size):
            # If we run out of samples in the middle of a batch, break
            if self._idx >= self.d_len:
                break

            # 1) Pick a class index from balanced_class_idx
            chosen_class_idx = self.balanced_class_idx[self._idx]
            chosen_label = self.ordered_clases[chosen_class_idx]

            # 2) Filter local indices whose label == chosen_label
            # ( local_idx is [0..d_len-1], but we need to map to `subset_indices`
            #   if we want the original dataset's index. )
            matching_local_idxs = [
                local_i for local_i, label in self.local_idx_to_label.items()
                if label == chosen_label
            ]

            # 3) Avoid picking the same local index repeatedly for this class
            last_local_idx = self.last_graph_idxs.get(chosen_label, None)
            while True:
                new_local_idx = random.choice(matching_local_idxs)
                if new_local_idx != last_local_idx:
                    self.last_graph_idxs[chosen_label] = new_local_idx
                    break

            # 4) Convert that local index to the actual dataset index
            dataset_idx = self.subset_indices[new_local_idx]

            # 5) Get (graph, y_raw_label) from dataset
            graph, y_raw = self.dataset[dataset_idx]

            # 6) (Optional) check consistency with chosen_label
            final_label = y_raw
            if self.pred_mode == "LUMINALSvsHER2vsTNBC" and final_label in ["Luminal A", "Luminal B"]:
                final_label = "Luminal"
            elif self.pred_mode == "OTHERvsTNBC" and final_label in ["Luminal A", "Luminal B", "HER2(+)"]:
                final_label = "Other"

            # Quick sanity check
            assert chosen_label == final_label, \
                f"Chosen label={chosen_label} but final_label={final_label}"

            # 7) One-hot encode the chosen_label
            Y = self.one_hot_encode(chosen_label)

            batch_graphs.append(graph)
            batch_labels.append(Y)

            self._idx += 1  # move forward in balanced_class_idx

        return batch_graphs, torch.tensor(batch_labels, dtype=torch.float32)

    def one_hot_encode(self, Y):
        """Convert label string -> one-hot vector based on pred_mode."""
        # Extend logic as needed
        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                if Y == 'Luminal A':
                    return [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    return [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    return [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    return [0., 0., 0., 1.]
            elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
                if Y == 'Luminal':
                    return [1., 0., 0.]
                if Y == 'HER2(+)':
                    return [0., 1., 0.]
                if Y == 'Triple negative':
                    return [0., 0., 1.]
            elif self.pred_mode == "OTHERvsTNBC":
                if Y == 'Other':
                    return [1., 0.]
                if Y == 'Triple negative':
                    return [0., 1.]
        # fallback
        return [Y]  # or raise an error


class MILDataGenerator_offline_graphs_balanced(object):
    def __len__(self):
        N = len(self.dataset) #.selected_graphs_paths)  # Use the length of the filtered graphs
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        if self.shuffle:
            #random.shuffle(self.dataset.selected_graphs_paths)
            random.shuffle(self.dataset.indices)

        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, graphs_on_ram, batch_size=1, shuffle=True, max_instances=100):
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset  # could be a Subset or the original dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        self.graphs_on_ram = graphs_on_ram
        self.d_len = len(self.dataset)  # Graphs filtered in the dataset class

        # Initialize iterator
        self._idx = 0
        self._reset()

        # Initialize last_graph_idxs dictionary to store last selected graph_idx for each class
        self.last_graph_idxs = {}

        # Distinguish between Subset and the original dataset
        if isinstance(dataset, Subset):
            self.parent_dataset = dataset.dataset
            self.subset_indices = dataset.indices
        else:
            self.parent_dataset = dataset
            self.subset_indices = range(len(self.parent_dataset))

        # Now you can access labels from the parent dataset
        # and gather them just for the indices in this subset.
        self.d_classes = [self.parent_dataset.labels[i] for i in self.subset_indices]


        # Extract labels directly from the dataset class
        # self.d_classes = self.dataset.labels

        # Adjust label filtering based on pred_mode
        if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
            self.d_classes = self.d_classes
        elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
            self.d_classes = ["Luminal" if x == "Luminal A" or x == "Luminal B" else x for x in self.d_classes]
        elif self.pred_mode == "OTHERvsTNBC":
            self.d_classes = ["Other" if x == "Luminal A" or x == "Luminal B" or x == "HER2(+)" else x for x in self.d_classes]

        # Dict with graph ids (keys) and corresponding labels (values)
        self.graphs_ids = np.arange(self.d_len)
        self.zip_gen = zip(self.graphs_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen)  # Dict with indexes of graph paths as keys, and classes as values.

        # Compute class weights for balancing
        self.ordered_clases, self.class_counts = np.unique(self.d_classes, return_counts=True)
        self.weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.d_classes), y=self.d_classes)
        self.weights_t = torch.DoubleTensor(list(self.weights))

        # Generate a list with the indexes of the balanced classes
        self.balanced_class_idx = torch.multinomial(input=self.weights_t, num_samples=self.d_len, replacement=True)

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        # Choose balanced class to find graph ID to return its images
        chosen_class_idx = self.balanced_class_idx[self._idx]
        chosen_label = self.ordered_clases[chosen_class_idx]

        # Graphs IDs with the chosen label
        matching_graph_idxs = [graph_idx for graph_idx, label in self.d_dict.items() if label == chosen_label]

        # Choose a new graph index that is different from the last one for this class
        last_graph_idx = self.last_graph_idxs.get(chosen_label)
        while True:
            # Randomly choose a graph index from the matching graph indexes
            graph_idx = random.choice(matching_graph_idxs)
            if graph_idx != last_graph_idx:
                self.last_graph_idxs[chosen_label] = graph_idx
                break

        # Get graph and label from the dataset
        graph, y = self.dataset.__getitem__(graph_idx)

        # Sanity check
        # if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
        #     assert chosen_label == y, "Chosen label does not match dataset label"
        # elif self.pred_mode == "LUMINALSvsHER2vsTNBC":
        #     if y == "Luminal A" or y == "Luminal B":
        #         y = "Luminal"
        #     assert chosen_label == y, "Chosen label does not match dataset label"
        # elif self.pred_mode == "OTHERvsTNBC":
        #     if y == "Luminal A" or y == "Luminal B" or y == "HER2(+)":
        #         y = "Other"
        #     assert chosen_label == y, "Chosen label does not match dataset label"

        # One-hot encoding of labels
        Y = chosen_label
        Y = self.one_hot_encode(Y)

        # Update bag index iterator
        self._idx += self.batch_size

        return graph, Y

    def one_hot_encode(self, Y):
        """Handles one-hot encoding for the various prediction columns."""
        if self.pred_column == "Molecular subtype":
            if self.pred_mode == "LUMINALAvsLAUMINALBvsHER2vsTNBC":
                if Y == 'Luminal A':
                    return [1., 0., 0., 0.]
                if Y == 'Luminal B':
                    return [0., 1., 0., 0.]
                if Y == 'HER2(+)':
                    return [0., 0., 1., 0.]
                if Y == 'Triple negative':
                    return [0., 0., 0., 1.]
            if self.pred_mode == "LUMINALSvsHER2vsTNBC":
                if Y == 'Luminal':
                    return [1., 0., 0.]
                if Y == 'HER2(+)':
                    return [0., 1., 0.]
                if Y == 'Triple negative':
                    return [0., 0., 1.]
            if self.pred_mode == "OTHERvsTNBC":
                if Y == 'Other':
                    return [1., 0.]
                if Y == 'Triple negative':
                    return [0., 1.]
        # Add more one-hot encoding conditions for other prediction columns as necessary
        return Y


import torch
import random
import numpy as np


class MILDataGenerator_offline_graphs(object):
    def __len__(self):
        N = len(self.dataset)  # Use the length of the filtered graphs
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def _reset(self):
        # Shuffle the subset indices instead of the dataset itself
        if self.shuffle:
            random.shuffle(self.indices)
        self._idx = 0

    def __init__(self, dataset, pred_column, pred_mode, graphs_on_ram, batch_size=1, shuffle=False, max_instances=100):
        self.pred_column = pred_column
        self.pred_mode = pred_mode
        self.dataset = dataset  # Dataset object
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_instances = max_instances
        self.graphs_on_ram = graphs_on_ram
        self.indices = list(range(len(self.dataset)))  # Store the indices of the subset
        self.d_len = len(self.indices)  # Use the length of the subset indices

        # Initialize iterator
        self._idx = 0
        self._reset()

        # Initialize last_graph_idxs dictionary to store last selected graph_idx for each class
        self.last_graph_idxs = {}

        # Extract labels directly from the dataset class
        self.d_classes = [self.dataset[i][1] for i in self.indices]  # Access labels via the dataset

        # Dict with graph ids (keys) and corresponding labels (values)
        self.graphs_ids = np.arange(self.d_len)
        self.zip_gen = zip(self.graphs_ids, self.d_classes)
        self.d_dict = dict(self.zip_gen)  # Dict with indexes of graph paths as keys, and classes as values.

    def __next__(self):
        # If dataset is completed, stop iterator
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        # Select a graph randomly or sequentially
        graph_idx = self._idx  # Sequential selection
        # Optionally, use random selection: graph_idx = random.choice(self.graphs_ids)

        # Get graph and label from the dataset
        graph, y = self.dataset.__getitem__(graph_idx)

        # One-hot encoding of labels
        Y = self.one_hot_encode(y)

        # Update the index iterator
        self._idx += self.batch_size

        return graph, Y

    def one_hot_encode(self, Y):
        """Handles one-hot encoding for the various prediction columns."""
        encoding_maps = {
            "Molecular subtype": {
                "LUMINALAvsLAUMINALBvsHER2vsTNBC": {
                    'Luminal A': [1., 0., 0., 0.],
                    'Luminal B': [0., 1., 0., 0.],
                    'HER2(+)': [0., 0., 1., 0.],
                    'Triple negative': [0., 0., 0., 1.]
                },
                "LUMINALSvsHER2vsTNBC": {
                    'Luminal': [1., 0., 0.],
                    'HER2(+)': [0., 1., 0.],
                    'Triple negative': [0., 0., 1.]
                },
                "OTHERvsTNBC": {
                    'Other': [1., 0.],
                    'Triple negative': [0., 1.]
                }
            }
        }

        # Handle the one-hot encoding based on prediction column and mode
        if self.pred_column in encoding_maps:
            if self.pred_mode in encoding_maps[self.pred_column]:
                return encoding_maps[self.pred_column][self.pred_mode].get(Y, None)

        return Y  # Return Y if no matching encoding is found
