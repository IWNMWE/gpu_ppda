import numpy as np
import random
import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from fedgraph.data_process import data_loader
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import attridict

def partition_graph(dataset : str, partition, n_parties, distribution_type = "average", iid_beta = 0.5, anchors=100):
        
    if partition == 'noniid' and dataset in ('cora', 'pubmed', 'citeseer'):


        config = {
            # Task, Method, and Dataset Settings
            "fedgraph_task": "NC",
            "dataset": dataset,
            "method": "FedGCN",  # Federated learning method, e.g., "FedGCN"
            "iid_beta": iid_beta,  # Dirichlet distribution parameter for label distribution among clients
            "distribution_type": "average",  # Distribution type among clients
            "n_trainer": n_parties,
            "batch_size": -1,  # -1 indicates full batch training
            "num_hops": 1,
            # Dataset Handling Options
            "use_huggingface": False,  # Load dataset directly from Hugging Face Hub
            "num_nodes_to_remove" : anchors,  # Number of nodes to remove for anchor selection
        }

        config = attridict(config)

        (
            edge_index,
            features,
            labels,
            idx_train,
            idx_test,
            class_num,
            split_node_indexes,
            communicate_node_global_indexes,
            in_com_train_node_local_indexes,
            in_com_test_node_local_indexes,
            global_edge_indexes_clients,
            val
        ) = data_loader(config)
        val_features, val_edge_index, val_labels, val_node_index = val
        # Initialize an empty list to store subgraphs
        client_subgraphs = []
        # Loop through clients
        for client_id in range(len(split_node_indexes)):
            # Get node and edge indices for this client
            node_subset = split_node_indexes[client_id]  # Nodes belonging to the client
            edge_subset = global_edge_indexes_clients[client_id]  # Edges for the client

            # Step 2: Filter nodes for this client (ensure all nodes in node_subset)
            sub_edge_index, _ = subgraph(
                subset=node_subset, edge_index= edge_subset, relabel_nodes=True, num_nodes= features.shape[0]
            )

            # Step 3: Extract features and labels for the client nodes
            sub_features = features[node_subset]
            sub_labels = labels[node_subset]
            num_test = 0.2 * sub_features.shape[0]
            test_mask = torch.zeros(sub_features.shape[0], dtype=torch.bool)
            test_mask[:int(num_test)] = 1
            # Store the subgraph
            client_subgraphs.append(Data(x=sub_features, edge_index=sub_edge_index, y=sub_labels))
            client_subgraphs[-1].test_mask = test_mask
        sub_val_edge_index, _ = subgraph(
            subset=val_node_index, edge_index=val_edge_index, relabel_nodes=True, num_nodes=features.shape[0] + val_features.shape[0]
        )
        val_graph  = Data(x=val_features, edge_index=sub_val_edge_index, y=val_labels)
        val_graph.test_mask = torch.ones(val_features.shape[0], dtype=torch.bool)
        return client_subgraphs, val_graph
    
    elif partition == 'noniid' and dataset == 'PPI':

        train_dataset = PPI(root="data/PPI", split="train")
        val_dataset = PPI(root="data/PPI", split="val")

        split_node_indexes, features, labels, global_edge_indexes_clients = data_loader(dataset, n_parties, distribution_type, iid_beta)

        # Initialize an empty list to store subgraphs
        client_subgraphs = []
        # Loop through clients
        for client_id in range(len(split_node_indexes)):
            # Get node and edge indices for this client
            node_subset = split_node_indexes[client_id]  # Nodes belonging 
            edge_subset = global_edge_indexes_clients[client_id]  # Edges f

            # Step 2: Filter nodes for this client (ensure all nodes in nod
            sub_edge_index, _ = subgraph(
                subset=node_subset, edge_index= edge_subset, relabel_nodes=True
            )

            # Step 3: Extract features and labels for the client nodes
            sub_features = features[node_subset]
            sub_labels = labels[node_subset]
            num_test = 0.1 * sub_features.shape[0]
            test_mask = torch.zeros(sub_features.shape[0], dtype=torch.bool)
            test_mask[:int(num_test)] = 1
            # Store the subgraph
            client_subgraphs.append(Data(x=sub_features, edge_index=sub_edge_index, y=sub_labels))
            client_subgraphs[-1].test_mask = test_mask
        return client_subgraphs