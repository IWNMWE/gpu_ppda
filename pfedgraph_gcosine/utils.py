import torch
import numpy as np
import copy
import cvxpy as cp
from torch_geometric.nn import GCNConv

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
        test_acc = float(test_correct.sum()) / float(data.test_mask.sum())
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

def build_initial_graph(E):

      n = E.shape[0]
      A = np.zeros((n,n))
      for i in range(n):
        max_value = np.max(E[i, :])
        den = n * max_value - np.sum(E[i, :])
        for j in range(n):
          A[i, j] = (max_value - E[i, j]) / den

      A = 0.5*(A + A.T)
      return A

def gen_graph_matrix(distance_matrix, assignment_matrix):
    
    
    A_dense = build_initial_graph(distance_matrix)
    A_sum = np.sum(A_dense, axis=1) 
    A_dense_norm  = A_dense / A_sum[:, np.newaxis]
    L = np.eye(A_dense.shape[0]) - A_dense_norm
    print(assignment_matrix.shape)
    L_graph = assignment_matrix.T @ L @ assignment_matrix
    D_graph = np.diag(np.diag(L_graph))
    A_graph = D_graph - L_graph
    A_graph = np.clip(A_graph, 0, None)
    A_sum = np.sum(A_graph, axis=1) 
    A_graph_norm  = A_graph / A_sum[:, np.newaxis]
    return torch.tensor(A_graph_norm)



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

