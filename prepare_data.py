from graphdataset import graphdataset_read
import numpy as np
import torch
from pfedgraph_gcosine.utils import FGLDataset

def compute_client_stats(data_list):

    # Find total number of classes across all clients
    num_classes = max([int(data.num_global_classes) for data in data_list]) + 1
    n_clients = len(data_list)

    traindata_cls_counts = np.zeros((n_clients, num_classes), dtype=int)

    for i, data in enumerate(data_list):
        train_nodes = data.train_mask.nonzero(as_tuple=True)[0]
        y_train = data.y[train_nodes].cpu().numpy()

        unq, unq_cnt = np.unique(y_train, return_counts=True)
        for c, cnt in zip(unq, unq_cnt):
            traindata_cls_counts[i, c] = cnt

    # normalize to get distributions
    data_distributions = traindata_cls_counts / traindata_cls_counts.sum(axis=1, keepdims=True)

    return traindata_cls_counts, data_distributions

def graphdataset_read(args):
    data = FGLDataset(args)
    local_data = data.local_data
    traindata_cls_counts, data_distributions = compute_client_stats(local_data)
    graph_matrix = data.get_graph_matrix()
    return local_data, graph_matrix, traindata_cls_counts, data_distributions
