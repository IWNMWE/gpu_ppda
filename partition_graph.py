import numpy as np
import random
import torch
from torch_geometric.utils import subgraph
from torch_geometric.data import Data
from fedgraph.data_process import data_loader, NC_load_data, label_dirichlet_partition, get_in_comm_indexes
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader
import attridict

def remove_nodes_for_val(features, adj, labels, idx_train, idx_val, idx_test, num_nodes_to_remove):
    row, col, edge_attr = adj.coo()
    edge_index = torch.stack([row, col], dim=0)
    
    num_nodes = features.shape[0]
    all_nodes = torch.arange(num_nodes)
    remove_nodes = torch.randperm(num_nodes)[:num_nodes_to_remove]
    
    keep_mask = ~torch.isin(all_nodes, remove_nodes)
    remove_mask = torch.isin(all_nodes, remove_nodes)

    # Mapping from old index to new index
    index_update = torch.full((num_nodes,), -1, dtype=torch.long)
    index_update[keep_mask] = torch.arange(keep_mask.sum())

    # Extract subgraph of removed nodes
    subgraph_mask = torch.isin(edge_index[0], remove_nodes) & torch.isin(edge_index[1], remove_nodes)
    subgraph_edge_index = edge_index[:, subgraph_mask]
    subgraph_features = features[remove_nodes]
    subgraph_labels = labels[remove_nodes]

    # Keep only remaining nodes
    new_features = features[keep_mask]
    new_labels = labels[keep_mask]

    # Keep edges between kept nodes
    keep_edges_mask = keep_mask[edge_index[0]] & keep_mask[edge_index[1]]
    edge_index = edge_index[:, keep_edges_mask]
    edge_attr = edge_attr[keep_edges_mask]

    # Reindex edges
    edge_index = index_update[edge_index]

    # Update idx_* by removing those that were removed
    idx_train = index_update[idx_train]
    idx_train = idx_train[idx_train != -1]
    idx_val = index_update[idx_val]
    idx_val = idx_val[idx_val != -1]
    idx_test = index_update[idx_test]
    idx_test = idx_test[idx_test != -1]

    return (new_features, edge_index[0], edge_index[1], edge_attr, new_labels, idx_train, idx_val, idx_test), (subgraph_features, subgraph_edge_index, subgraph_labels, remove_nodes)


def data_loader_NC(args: attridict) -> tuple:
    print("config: ", args)
    if not args.use_huggingface:
        # process on the server
        features, adj, labels, idx_train, idx_val, idx_test = NC_load_data(args.dataset)
        #print(idx_val, features.shape)
        train, val = remove_nodes_for_val(features, adj, labels, idx_train, idx_val, idx_test, args.num_nodes_to_remove)
        features, row, col, edge_attr, labels, idx_train, idx_val, idx_test = train
        class_num = labels.max().item() + 1
        #row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col], dim=0)
        #######################################################################
        # Split Graph for Federated Learning
        # ----------------------------------
        # FedGraph currents has two partition methods: label_dirichlet_partition
        # and community_partition_non_iid to split the large graph into multiple trainers
        #split data indexes for each trainer(Global)
        split_node_indexes = label_dirichlet_partition(
            labels,
            len(labels),
            class_num,
            args.n_trainer,
            beta=args.iid_beta,
            distribution_type=args.distribution_type,
        )

        for i in range(args.n_trainer):
            split_node_indexes[i] = np.array(split_node_indexes[i])
            split_node_indexes[i].sort()
            split_node_indexes[i] = torch.tensor(split_node_indexes[i])

        (
            communicate_node_global_indexes,
            in_com_train_node_local_indexes,
            in_com_test_node_local_indexes,
            global_edge_indexes_clients,
        ) = get_in_comm_indexes(
            edge_index,
            split_node_indexes,
            args.n_trainer,
            args.num_hops,
            idx_train,
            idx_test,
        )
    return (
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
    )

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
            "num_hops": 0,
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
        ) = data_loader_NC(config)
        val_features, val_edge_index, val_labels, val_node_index = val
        # Initialize an empty list to store subgraphs
        client_subgraphs = []
        # Loop through clients
        for client_id in range(len(split_node_indexes)):
            # Get node and edge indices for this client
            client_mask = torch.isin(communicate_node_global_indexes[client_id], split_node_indexes[client_id])
            node_subset = communicate_node_global_indexes[client_id]  # Nodes belonging to the client
            edge_subset = global_edge_indexes_clients[client_id]  # Edges for the client
            print('hello ji')
            print(node_subset.shape)

            # Step 2: Filter nodes for this client (ensure all nodes in node_subset)
            sub_edge_index, _ = subgraph(
                subset=node_subset, edge_index= edge_subset, relabel_nodes=True, num_nodes= features.shape[0]
            )

            # Step 3: Extract features and labels for the client nodes
            sub_features = features[node_subset]
            sub_labels = labels[node_subset]
            train_mask = torch.zeros(sub_features.shape[0], dtype=torch.bool)
            test_mask = torch.zeros(sub_features.shape[0], dtype=torch.bool)
            train_mask[in_com_train_node_local_indexes[client_id]] = True
            test_mask[in_com_test_node_local_indexes[client_id]] = True
            # Store the subgraph
            client_subgraphs.append(Data(x=sub_features, edge_index=sub_edge_index, y=sub_labels, test_mask=test_mask, train_mask=train_mask, client_mask=client_mask))
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