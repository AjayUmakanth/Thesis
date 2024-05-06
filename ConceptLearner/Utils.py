import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
from datetime import datetime
import os


def k_hop_subgraph(start_node: int, start_label: str, data: HeteroData, hops: int) -> HeteroData:
    """
    Generates a subgraph of a given graph by using the k-hop technique. This means that the subgraph will contain any
    nodes which are reachable by a path of length k or shorter.

    Args:
        start_node: The index of the node the algorithm uses as a starting point.
        start_label: The type of the starting node.
        data: The graph which will be used for sampling.
        hops: The number of hops.

    Returns:
        HeteroData: The subgraph generated by the k-hop algorithm.

    """
    node_types = data.node_types
    edge_types = data.edge_types

    if start_label not in node_types:
        raise Exception()

    edges = dict()

    for i in data.edge_items():
        edges[i[0]] = i[1]['edge_index'].numpy()

    result = dict()
    mapping = dict()
    for i in node_types:
        result[i] = dict()
        mapping[i] = list()

    for i in edge_types:
        result[i] = dict()

    # Include the start point
    result[start_label]["x"] = np.array([np.append([0, ], data[start_label]["x"][start_node].numpy()), ])
    mapping[start_label] = [start_node, ]

    if "y" in data[start_label].keys():
        result[start_label]["y"] = np.array([data[start_label]["y"][start_node], ])

    for _ in range(hops):
        loop_result = dict()
        for i in node_types:
            loop_result[i] = dict()
        for i in edge_types:
            loop_result[i] = dict()

        # Inspect every possible node type
        for ii in node_types:
            if result[ii]:
                for iii in range(len(result[ii]["x"])):

                    if result[ii]["x"][iii][0] != 1:
                        # Set the current node to processed
                        result[ii]["x"][iii][0] = 1

                        # Set the original index of the currently processed node
                        current_node = mapping[ii][iii]

                        # Look for all edges which start in the current node
                        for edge_label in edges.keys():
                            if edge_label[0] == ii:
                                for edge in zip(edges[edge_label][0], edges[edge_label][1]):
                                    if edge[0] == current_node:
                                        # An edge which starts in the given node is found
                                        if edge[1] not in mapping[edge_label[2]]:
                                            node_x = data[edge_label[2]]['x'][edge[1]]
                                            node_y = None

                                            if "y" in data[edge_label[2]].keys():
                                                node_y = data[edge_label[2]]['y'][edge[1]]

                                            if not loop_result[edge_label[2]]:
                                                # If the dictionary did not exist
                                                loop_result[edge_label[2]]['x'] = \
                                                    np.array([np.append([0, ], node_x.numpy()), ])
                                                if node_y is not None:
                                                    loop_result[edge_label[2]]['y'] = np.array([node_y, ])

                                            else:
                                                # If the dictionary already exists
                                                loop_result[edge_label[2]]['x'] = \
                                                    np.append(loop_result[edge_label[2]]['x'],
                                                              np.array([np.append([0, ], node_x.numpy()), ]), axis=0)
                                                if node_y is not None:
                                                    loop_result[edge_label[2]]['y'] = np.append(
                                                        loop_result[edge_label[2]]['y'], node_y)

                                            mapping[edge_label[2]] = mapping[edge_label[2]] + [edge[1], ]

                                        if result[edge_label]:
                                            result[edge_label]['edge_index'] = np.append(
                                                result[edge_label]['edge_index'],
                                                [[len(mapping[edge_label[0]]) - 1, ],
                                                 [len(mapping[edge_label[2]]) - 1, ]], axis=1)
                                        else:
                                            result[edge_label]['edge_index'] = np.array(
                                                [[len(mapping[edge_label[0]]) - 1, ],
                                                 [len(mapping[edge_label[2]]) - 1, ]])

        for i in node_types:
            if loop_result[i]:
                if result[i]:
                    result[i]["x"] = np.append(result[i]["x"], loop_result[i]["x"], axis=0)
                else:
                    result[i]["x"] = loop_result[i]["x"]

    # Change the subgraph to the needed format
    for i in node_types:
        if result[i]:
            result[i]["x"] = torch.from_numpy(np.delete(result[i]["x"], 0, axis=1))
    for i in edge_types:
        if result[i]:
            result[i]["edge_index"] = torch.from_numpy(result[i]["edge_index"])

    return HeteroData(result)


def probability_edge_sampling(data: HeteroData, alpha: float, edge_types: list[str] = None) -> HeteroData:
    """
    Generates a subgraph of a given graph by using the probability edge sampling technique. This means that every edge
    has a probability of alpha to be included in the subgraph.

    Args:
        data: The graph which will be used for sampling.
        alpha: The probability with which an edge will be included in the subgraph.
        edge_types: A list of edge types which will be included in the subgraph. If None all edge types will be
        included.

    Returns:
        HeteroData: The subgraph generated by the probability edge sampling algorithm.

    """
    if edge_types is None:
        edge_types = list()
    subgraph = data.clone()
    if len(edge_types) == 0:
        edge_types = data.edge_types

    if not all(elem in data.edge_types for elem in edge_types):
        raise Exception()

    if alpha < 0 or alpha > 1:
        raise Exception()

    for i in edge_types:
        origin = data[i]['edge_index'][0]
        dest = data[i]['edge_index'][1]

        choice = np.random.choice([True, False], size=len(origin), p=[alpha, 1 - alpha])

        subgraph[i]['edge_index'] = torch.from_numpy(np.array([origin[choice], dest[choice]]))

    return subgraph


def visualize_hetero_data(hetero_data: HeteroData, predicate_only: bool = True) -> None:
    """
    Visualizes a HeteroData object using NetworkX and Matplotlib.

    Args:
    - hetero_data (HeteroData): The input HeteroData object to be visualized.
    - predicate_only (bool): If True, display edge labels with the second part of the type.

    Returns:
    - None

    Steps:
    1. Convert HeteroData to NetworkX graph.
    2. Draw the graph with node types and edge types using NetworkX and Matplotlib.
    """

    # Step 1: Convert HeteroData to NetworkX graph
    G = to_networkx(hetero_data)

    # Step 2: Draw the graph with node types and edge types
    plt.figure(figsize=(20, 20))

    # Generate layout for nodes
    pos = nx.spring_layout(G)

    # Extract node types
    node_types = [data[1]['type'] for data in G.nodes(data=True)]

    # Define a list of 16 colors for node types
    node_color_list = [
        'skyblue', 'lightgreen', 'salmon', 'gold',
        'cyan', 'magenta', 'orange', 'pink',
        'yellow', 'blue', 'green', 'red',
        'purple', 'brown', 'teal', 'lavender'
    ]

    # Remove duplicates to get unique node types
    unique_node_types = list(dict.fromkeys(node_types))

    # Create a mapping of node types to colors
    node_colors = dict(zip(unique_node_types, node_color_list[:len(unique_node_types)]))

    # Assign node colors based on node types
    node_colors = [node_colors[node_type] for node_type in node_types]

    # Draw nodes with specific colors based on node types
    nx.draw(G, pos, with_labels=True, node_size=300, node_color=node_colors, font_weight='bold')

    # Add edge labels for edge types
    if predicate_only:
        # Extract edge labels based on the middle part of edge types
        edge_labels = {(u, v): d['type'][1] for u, v, d in G.edges(data=True)}
    else:
        # Extract complete edge types as labels
        edge_labels = {(u, v): d['type'] for u, v, d in G.edges(data=True)}

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Set the plot title
    plt.title('Visualization of Heterogeneous Graph with Node and Edge Types')

    # Display the plot
    plt.show(block=True)


def sample_data_generator(data: HeteroData, batch_size: int = 256, number_of_input_nodes: int = 24,
                          number_of_neighbors: int = 5, iterations: int = 2):
    kwargs = {'num_workers': 1, 'persistent_workers': True}  # current kwargs for convenience remove later

    # Identify the class associated with 'y' labels in the provided HeteroData
    _class_of_node = _find_classes_with_y_labels(data)

    # Prepare input nodes for training by selecting the specific class and its associated training mask
    train_input_nodes = (
        _class_of_node, data[_class_of_node].train_mask[:number_of_input_nodes])  # Limit input nodes to a set number

    # Create and return a NeighborLoader to process the HeteroData
    # This loader is configured with specific parameters:
    # - num_neighbors: Number of neighbors for each node in each iteration
    # - shuffle: Flag indicating whether to shuffle the data (set to False)
    # - input_nodes: Nodes to be used for training, limited to a specified number (number_of_input_nodes)
    # - batch_size: Number of samples in each mini-batch
    # - **kwargs: Additional keyword arguments for DataLoader configuration (default or to be modified)
    loader = NeighborLoader(data, num_neighbors=[number_of_neighbors] * iterations,
                            shuffle=True,  # No shuffling of data
                            input_nodes=train_input_nodes,
                            batch_size=batch_size,
                            **kwargs)
    return iter(loader)  # Utilize DataLoader with provided and default settings


def _find_classes_with_y_labels(data: HeteroData = None, first_only=True):
    # Finding class with 'y' labels
    if data is None:
        return False
    classes_with_y_labels = [key for key in data.to_dict().keys() if 'y' in data.to_dict().get(key).keys()]
    if first_only:
        return classes_with_y_labels[0]
    return classes_with_y_labels


def save_hgnn_model(hgnn=None):
    """
    Save the current state of the model.

    Args:
        hgnn: An instance of the model class (e.g., a PyTorch model) whose state dictionary
              will be saved.

    Returns:
        str: The name of the saved model file.

    Notes:
        If `hgnn` is None, the function prints a message and returns without saving anything.
    """
    if hgnn:
        print('-------------------------------------------------------------------')
        now = datetime.now()
        current_time = now.strftime("%d.%m.%Y_%H-%M-%S")
        modelname = "model_" + current_time

        # Create the "models" directory if it doesn't exist
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        print(f'Saving the current model under: {modelname}...')
        torch.save(hgnn.model.state_dict(), os.path.join(models_dir, modelname))
        print('-------------------------------------------------------------------\n')
        return modelname


def load_hgnn_model(hgnn=None, state_to_load: str = None):
    """
    Load a specific state of the model.

    Args:
        hgnn: An instance of the model class (e.g., a PyTorch model) where the state dictionary
              will be loaded.
        state_to_load (str): The name of the state dictionary file to load.

    Returns:
        str: The file path of the loaded model.

    Notes:
        If `hgnn` or `state_to_load` is None, the function prints a message and returns without loading anything.

    Raises:
        FileNotFoundError: If the specified state dictionary file does not exist.
    """
    if hgnn is None or state_to_load is None:
        print('Specifying the model and state which should be loaded is mandatory!')
        return

    try:
        print('-------------------------------------------------------------------')
        print(f'Loading the model: {state_to_load}...')
        hgnn.model.load_state_dict(torch.load(os.path.join("models", state_to_load)))
        print('-------------------------------------------------------------------\n')
        return os.path.join("models", state_to_load)
    except FileNotFoundError:
        print(f"No such file: {state_to_load}.")
        print('-------------------------------------------------------------------\n')
        return


def load_latest_hgnn_model(hgnn=None):
    """
    Load the latest model from the 'models' directory.

    Args:
        hgnn: Optional parameter. An instance of the model class (e.g., a PyTorch model) where the state dictionary
              will be loaded. If not provided, this parameter can be ignored.

    Returns:
        str: The file path of the loaded model.

    Raises:
        FileNotFoundError: If no model files are found in the 'models' directory.
    """
    if hgnn:
        # Find all files in the models directory and get the file with the latest creation time
        list_of_files = glob.glob(os.path.join('models', '*'))

        if not list_of_files:
            raise FileNotFoundError("No model files found in the 'models' directory.")

        latest_file = max(list_of_files, key=os.path.getctime)
        print('-------------------------------------------------------------------')
        print(f'Loading the latest model: {latest_file}...')

        # Get the name of the latest file and load its dictionary
        model_name = os.path.basename(latest_file)
        hgnn.model.load_state_dict(torch.load(os.path.join("models", model_name)))
        print('-------------------------------------------------------------------\n')
        return os.path.join("models", model_name)
