import torch
import numpy as np
import copy
import cvxpy as cp
from torch_geometric.nn import GCNConv

import os
from os import path as osp
import torch_geometric
from openfgl.data.global_dataset_loader import load_global_dataset
from torch_geometric.data import Dataset, Data
from openfgl.data.distributed_dataset_loader import FGLDataset
from torch_geometric.data import InMemoryDataset, Data
from openfgl.utils.basic_utils import extract_floats, idx_to_mask_tensor, mask_tensor_to_idx


def compute_local_test_accuracy(model, data, data_distribution):

    model.eval()

    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        out = model(data.x.to('cuda'), data.edge_index.to('cuda'))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask].to('cuda')  # Check against ground-truth labels.
        # test_acc = float(test_correct.sum()) / float(data.test_mask.sum())
        generalized_total = data.test_mask.sum() 
        generalized_correct = test_correct.sum()
        labels = data.y[data.test_mask] 
        for i in range(labels.shape[0]):
            true_label = labels[i]
            total_label_num[true_label] += 1
            if test_correct[i]:
                correct_label_num[true_label] += 1
    personalized_correct = (correct_label_num * data_distribution).sum()
    personalized_total = (total_label_num * data_distribution).sum()
    
    model.to('cpu')
    return personalized_correct / personalized_total, generalized_correct / generalized_total

def compute_local_val_accuracy(model, data, data_distribution):

    model.eval()

    total_label_num = np.zeros(len(data_distribution))
    correct_label_num = np.zeros(len(data_distribution))
    model.cuda()
    generalized_total, generalized_correct = 0, 0
    with torch.no_grad():
        out = model(data.x.to('cuda'), data.edge_index.to('cuda'))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.val_mask] == data.y[data.val_mask].to('cuda')  # Check against ground-truth labels.
        # test_acc = float(test_correct.sum()) / float(data.test_mask.sum())
        generalized_total = data.val_mask.sum() 
        generalized_correct = test_correct.sum()
    
    model.to('cpu')
    return generalized_correct / generalized_total


def cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric):
    model_similarity_matrix = torch.zeros((len(nets_this_round),len(nets_this_round)))
    index_clientid = list(nets_this_round.keys())
    for i in range(len(nets_this_round)):
        model_i = nets_this_round[index_clientid[i]].state_dict()
        for key in dw[index_clientid[i]]:
            dw[index_clientid[i]][key] =  model_i[key] - initial_global_parameters[key]
    for i in range(len(nets_this_round)):
        for j in range(i, len(nets_this_round)):
            if similarity_matric == "all":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten_all(dw[index_clientid[i]]).unsqueeze(0), weight_flatten_all(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff
            elif  similarity_matric == "fc":
                diff = - torch.nn.functional.cosine_similarity(weight_flatten(dw[index_clientid[i]]).unsqueeze(0), weight_flatten(dw[index_clientid[j]]).unsqueeze(0))
                if diff < - 0.9:
                    diff = - 1.0
                model_similarity_matrix[i, j] = diff
                model_similarity_matrix[j, i] = diff

    # print("model_similarity_matrix" ,model_similarity_matrix)
    return model_similarity_matrix

def update_graph_matrix_neighbor(graph_matrix, nets_this_round, initial_global_parameters, dw, fed_avg_freqs, lambda_1, similarity_matric):
    # index_clientid = torch.tensor(list(map(int, list(nets_this_round.keys()))))     # for example, client 'index_clientid[0]'s model difference vector is model_difference_matrix[0] 
    index_clientid = list(nets_this_round.keys())
    # model_difference_matrix = cal_model_difference(index_clientid, nets_this_round, nets_param_start, difference_measure)
    model_difference_matrix = cal_model_cosine_difference(nets_this_round, initial_global_parameters, dw, similarity_matric)
    graph_matrix = optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lambda_1, fed_avg_freqs)
    # print(f'Model difference: {model_difference_matrix[0]}')
    # print(f'Graph matrix: {graph_matrix}')
    return graph_matrix


def optimizing_graph_matrix_neighbor(graph_matrix, index_clientid, model_difference_matrix, lamba, fed_avg_freqs):
    n = model_difference_matrix.shape[0]
    p = np.array(list(fed_avg_freqs.values()))
    P = lamba * np.identity(n)
    P = cp.atoms.affine.wraps.psd_wrap(P)
    G = - np.identity(n)
    h = np.zeros(n)
    A = np.ones((1, n))
    b = np.ones(1)
    for i in range(model_difference_matrix.shape[0]):
        model_difference_vector = model_difference_matrix[i]
        d = model_difference_vector.numpy()
        q = d - 2 * lamba * p
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q.T @ x),
                  [G @ x <= h,
                   A @ x == b]
                  )
        prob.solve()

        graph_matrix[index_clientid[i], index_clientid] = torch.Tensor(x.value)
    return graph_matrix

def aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_w):
    tmp_client_state_dict = {}
    cluster_model_vectors = {}
    for client_id in nets_this_round.keys():
        tmp_client_state_dict[client_id] = copy.deepcopy(global_w)
        cluster_model_vectors[client_id] = torch.zeros_like(weight_flatten_all(global_w))
        for key in tmp_client_state_dict[client_id]:
            tmp_client_state_dict[client_id][key] = torch.zeros_like(tmp_client_state_dict[client_id][key])

    for client_id in nets_this_round.keys():
        tmp_client_state = tmp_client_state_dict[client_id]
        cluster_model_state = cluster_model_vectors[client_id]
        aggregation_weight_vector = graph_matrix[client_id]

        # if client_id==0:
        #     print(f'Aggregation weight: {aggregation_weight_vector}. Summation: {aggregation_weight_vector.sum()}')
        
        for neighbor_id in nets_this_round.keys():
            net_para = nets_this_round[neighbor_id].state_dict()
            for key in tmp_client_state:
                tmp_client_state[key] += net_para[key] * aggregation_weight_vector[neighbor_id]

        for neighbor_id in nets_this_round.keys():
            net_para = weight_flatten_all(nets_this_round[neighbor_id].state_dict())
            cluster_model_state += net_para * (aggregation_weight_vector[neighbor_id] / torch.linalg.norm(net_para))
               
    for client_id in nets_this_round.keys():
        nets_this_round[client_id].load_state_dict(tmp_client_state_dict[client_id])
    
    return cluster_model_vectors

def weight_flatten(model):
    params = []
    for k in model:
        if 'lin' in k:
            params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def weight_flatten_all(model):
    params = []
    for k in model:
        params.append(model[k].reshape(-1))
    params = torch.cat(params)
    return params

def compute_acc(net, data):

    net.to(torch.device('cuda'))
    net.eval()
    with torch.no_grad():
        out = net(data.x.to('cuda'), data.edge_index.to('cuda'))
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        test_correct = pred[data.test_mask] == data.y[data.test_mask].to('cuda')  # Check against ground-truth labels.
        test_acc = float(test_correct.sum()) / float(data.test_mask.sum())  # Derive ratio of correct predictions.
    net.to('cpu')
    return test_acc

def compute_loss(net, data):

    criterion = torch.nn.CrossEntropyLoss()
    net.eval()
    loss = 0
    net.cuda()
    with torch.no_grad():
      out = net(data.x.to('cuda'), data.edge_index.to('cuda'))  # Perform a single forward pass.
      loss = criterion(out[data.test_mask], data.y[data.test_mask].to('cuda'))  
    net.to('cpu')
    return loss


class MyDataset(InMemoryDataset):
    def __init__(self, data, transform=None, pre_transform=None):
        super().__init__('.', transform, pre_transform)
        self.data, self.slices = self.collate([data])

    def __len__(self):
        return 1

    def get(self, idx):
        return self.data

def local_subgraph_train_val_test_split(local_subgraph, split, shuffle=True):
        """
        Split the local subgraph into train, validation, and test sets.

        Args:
            local_subgraph (object): Local subgraph to be split.
            split (str or tuple): Split ratios or default split identifier.
            shuffle (bool, optional): If True, shuffle the subgraph before splitting. Defaults to True.

        Returns:
            tuple: Masks for the train, validation, and test sets.
        """
        num_nodes = local_subgraph.x.shape[0]

        if split == "default_split":
            train_, val_, test_ = 0.2, 0.4, 0.4
        else:
            train_, val_, test_ = extract_floats(split)

        train_mask = idx_to_mask_tensor([], num_nodes)
        val_mask = idx_to_mask_tensor([], num_nodes)
        test_mask = idx_to_mask_tensor([], num_nodes)
        for class_i in range(local_subgraph.num_global_classes):
            class_i_node_mask = local_subgraph.y == class_i
            num_class_i_nodes = class_i_node_mask.sum()

            class_i_node_list = mask_tensor_to_idx(class_i_node_mask)
            if shuffle:
                np.random.shuffle(class_i_node_list)
            train_mask += idx_to_mask_tensor(class_i_node_list[:int(train_ * num_class_i_nodes)], num_nodes)
            val_mask += idx_to_mask_tensor(class_i_node_list[int(train_ * num_class_i_nodes) : int((train_+val_) * num_class_i_nodes)], num_nodes)
            test_mask += idx_to_mask_tensor(class_i_node_list[int((train_+val_) * num_class_i_nodes): min(num_class_i_nodes, int((train_+val_+test_) * num_class_i_nodes))], num_nodes)


        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()
        return train_mask, val_mask, test_mask

class PFGDataset(FGLDataset):
    def __init__(self, args, num_anchors=100, anchor_seed=42, **kwargs):

        self.num_anchors = num_anchors
        self.anchor_seed = anchor_seed
        self.X_a = None
        super().__init__(args, **kwargs)
        self.anchor_global_ids = []

    def process(self):
        """Process the dataset according to the specified simulation mode."""

        global_dataset = load_global_dataset(self.global_root, scenario=self.args.scenario, dataset=self.args.dataset[0])
        print(global_dataset[0])
        original_num_nodes = global_dataset.x.shape[0]
        anchor_ids = self._select_anchor_nodes(global_dataset, self.num_anchors, original_num_nodes)

        self.X_a = global_dataset.data.x[anchor_ids]
        self.y_a = global_dataset.data.y[anchor_ids]
        self.anchor_global_ids = anchor_ids
        modified_global_dataset = self._remove_nodes_from_dataset(global_dataset, anchor_ids, original_num_nodes)
        modified_global_dataset = MyDataset(modified_global_dataset)



        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        if self.args.simulation_mode == "graph_fl_label_skew":
            from openfgl.data.simulation import graph_fl_label_skew
            self.local_data = graph_fl_label_skew(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "graph_fl_cross_domain":
            from openfgl.data.simulation import graph_fl_cross_domain
            self.local_data = graph_fl_cross_domain(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "graph_fl_topology_skew":
            from openfgl.data.simulation import graph_fl_topology_skew
            self.local_data = graph_fl_topology_skew(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_label_skew":
            from openfgl.data.simulation import subgraph_fl_label_skew
            self.local_data = subgraph_fl_label_skew(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain_plus":
            from openfgl.data.simulation import subgraph_fl_louvain_plus
            self.local_data = subgraph_fl_louvain_plus(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis_plus":
            from openfgl.data.simulation import subgraph_fl_metis_plus
            self.local_data = subgraph_fl_metis_plus(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_louvain":
            from openfgl.data.simulation import subgraph_fl_louvain
            self.local_data = subgraph_fl_louvain(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "subgraph_fl_metis":
            from openfgl.data.simulation import subgraph_fl_metis
            self.local_data = subgraph_fl_metis(self.args, modified_global_dataset)
        elif self.args.simulation_mode == "graph_fl_feature_skew":
            from openfgl.data.simulation import graph_fl_feature_skew
            self.local_data = graph_fl_feature_skew(self.args, modified_global_dataset)

        

        for client_id in range(self.args.num_clients):
            train_mask, val_mask, test_mask = local_subgraph_train_val_test_split(self.local_data[client_id], self.args.train_val_test)
            self.local_data[client_id].train_mask = train_mask
            self.local_data[client_id].test_mask = test_mask
            self.local_data[client_id].val_mask = val_mask
            self.save_client_data(self.local_data[client_id], client_id)

        self.save_dataset_description()
        self._save_anchor_data()
        print("Data creation complete")

    def _select_anchor_nodes(self, dataset, num_anchors, num_nodes):
        """Select anchor nodes from the global dataset."""
        torch.manual_seed(self.anchor_seed)
        anchor_ids = torch.randperm(num_nodes)[:num_anchors]
        return anchor_ids

    def _remove_nodes_from_dataset(self, dataset, anchor_ids, num_nodes):
        """Remove anchor nodes from the global dataset."""
        from torch_geometric.utils import subgraph, mask_to_index
        keep_mask = torch.ones(num_nodes, dtype=torch.bool)
        keep_mask[anchor_ids] = False
        keep_indices = mask_to_index(keep_mask)

        edge_index, edge_attr = subgraph(keep_indices, dataset.edge_index, edge_attr = getattr(dataset, 'edge_attr', None), relabel_nodes=True, num_nodes=num_nodes)
        modified_data = Data(
            x = dataset.x[keep_indices],
            edge_index = edge_index,
            y = dataset.y[keep_indices],
        )
        if edge_attr is not None:
            modified_data.edge_attr = edge_attr[edge_index[0]]

        for key, value in dataset[0].items():
            if key not in ['edge_index', 'edge_attr', 'x', 'y', 'num_nodes']:
                try:
                  if self._is_mask_attr(key, value, num_nodes):
                    updated_mask = value[keep_indices]
                    setattr(modified_data, key, updated_mask)
                  else:
                    setattr(modified_data, key, value)
                except Exception:
                  pass


        return modified_data

    def _is_mask_attr(self, key, value, num_nodes):
        """Check if a mask attribute is valid."""

        mask_sub = ['_mask', 'mask_', 'train', 'val', 'test']
        name_suggests_mask = any(keyword in key.lower()  for keyword in mask_sub)
        is_tensor = isinstance(value, torch.Tensor)

        if not is_tensor:
          return False

        correct_size = value.numel() == num_nodes
        is_boolean_or_binary = (value.dtype == torch.bool) or (value.dtype in [torch.int, torch.long] and torch.all((value == 0) | (value == 1)))
        is_1d = value.dim() == 1
        is_node_mask = (name_suggests_mask and correct_size and is_1d and is_boolean_or_binary)
        return is_node_mask

    def _save_anchor_data(self):
        anchor_dir = os.path.join(self.processed_dir, "anchor_data")
        os.makedirs(anchor_dir, exist_ok=True)
        torch.save(self.X_a, os.path.join(anchor_dir, "X_a.pt"))
        torch.save(self.y_a, os.path.join(anchor_dir, "y_a.pt"))

    def get_anchors(self):
        anchor_dir = os.path.join(self.processed_dir, "anchor_data")
        os.makedirs(anchor_dir, exist_ok=True)
        self.X_a = torch.load(os.path.join(anchor_dir, "X_a.pt"))
        self.y_a = torch.load(os.path.join(anchor_dir, "y_a.pt"))
        return self.X_a, self.y_a
    
    def get_graph_matrix(self):
        anchor_dir = os.path.join(self.processed_dir, "anchor_data")
        os.makedirs(anchor_dir, exist_ok=True)
        self.graph_matrix = torch.load(os.path.join(anchor_dir, "graph_matrix.pt"))
        return self.graph_matrix
    
    def set_graph_matrix(self, graph_matrix):
        anchor_dir = os.path.join(self.processed_dir, "anchor_data")
        os.makedirs(anchor_dir, exist_ok=True)
        torch.save(graph_matrix, os.path.join(anchor_dir, "graph_matrix.pt"))