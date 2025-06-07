import copy
import math
import random
import time

import numpy as np
import torch
import torch.optim as optim

from pfedgraph_gcosine.config import get_args
from torch_geometric.utils import subgraph
from pfedgraph_gcosine.utils import aggregation_by_graph, update_graph_matrix_neighbor, compute_acc, compute_local_test_accuracy, gen_graph_matrix
from prepare_data import get_dataloader
from attack import *
from model import coragcn, niidgcn


def local_train_pfedgraph(args, round, nets_this_round, cluster_models, datasets, data_distributions, val, best_val_acc_list, best_test_acc_list, benign_client_list):
    
    for net_id, net in nets_this_round.items():
        
        data_distribution = data_distributions[net_id]

        if net_id in benign_client_list:
            val_acc = compute_acc(net, val)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, datasets[net_id], data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f} | Anchor Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc, val_acc))

        # Set Optimizer
        if args.optimizer == 'adam':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
        elif args.optimizer == 'amsgrad':
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg,
                                amsgrad=True)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=0.9, weight_decay=args.reg)
        criterion = torch.nn.CrossEntropyLoss()
        if round > 0:
            cluster_model = cluster_models[net_id].cuda()
        
        net.cuda()
        net.train()
        for iteration in range(args.num_local_iterations):
            
            x = datasets[net_id].x
            target = datasets[net_id].y 
            x, target = x.cuda(), target.cuda()
            
            optimizer.zero_grad()

            out = net(datasets[net_id].x.to('cuda'), datasets[net_id].edge_index.to('cuda'))
            loss = criterion(out[datasets[net_id].test_mask == 0], target[datasets[net_id].test_mask == 0])
        

            if round > 0:
                flatten_model = []
                for param in net.parameters():
                    flatten_model.append(param.reshape(-1))
                flatten_model = torch.cat(flatten_model)
                loss2 = args.lam * torch.dot(cluster_model, flatten_model) / torch.linalg.norm(flatten_model)
                loss2.backward()
                
            loss.backward()
            optimizer.step()
        
        if net_id in benign_client_list:
            val_acc = compute_acc(net, val)
            personalized_test_acc, generalized_test_acc = compute_local_test_accuracy(net, datasets[net_id], data_distribution)

            if val_acc > best_val_acc_list[net_id]:
                best_val_acc_list[net_id] = val_acc
                best_test_acc_list[net_id] = personalized_test_acc
            print('>> Client {} test1 | (Pre) Personalized Test Acc: ({:.5f}) | Generalized Test Acc: {:.5f} | Anchor Test Acc: {:.5f}'.format(net_id, personalized_test_acc, generalized_test_acc, val_acc))
        net.to('cpu')
    return np.array(best_test_acc_list)[np.array(benign_client_list)].mean()


args, cfg = get_args()
print(args)
seed = args.init_seed
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
random.seed(seed)

n_party_per_round = int(args.n_parties * args.sample_fraction)
party_list = [i for i in range(args.n_parties)]
party_list_rounds = []
if n_party_per_round != args.n_parties:
    for i in range(args.comm_round):
        party_list_rounds.append(random.sample(party_list, n_party_per_round))
else:
    for i in range(args.comm_round):
        party_list_rounds.append(party_list)

benign_client_list = random.sample(party_list, int(args.n_parties * (1-args.attack_ratio)))
benign_client_list.sort()
print(f'>> -------- Benign clients: {benign_client_list} --------')

datasets, traindata_cls_counts, data_distributions, val = get_dataloader(args,cfg)
if args.dataset in ('cora', 'pubmed', 'citeseer'):
    model = coragcn
elif args.dataset == 'niid':
    model = niidgcn
    
global_model = model(cfg['classes_size'], cfg['feature_size'])
global_parameters = global_model.state_dict()
local_models = []
best_val_acc_list, best_test_acc_list = [],[]
dw = []
for i in range(cfg['client_num']):
    local_models.append(model(cfg['classes_size'], cfg['feature_size']))
    dw.append({key : torch.zeros_like(value) for key, value in local_models[i].named_parameters()})
    best_val_acc_list.append(0)
    best_test_acc_list.append(0)

graph_matrix = torch.ones(len(local_models), len(local_models)) / (len(local_models)-1)                 # Collaboration Graph
graph_matrix[range(len(local_models)), range(len(local_models))] = 0

for net in local_models:
    net.load_state_dict(global_parameters)

    
cluster_model_vectors = {}

if args.ppda:
    distance_matrix = np.load(args.load_graph_path + "D_esti.npy")
    assignement_matrix = np.load(args.load_graph_path + "C_new_500.npy")
    graph_matrix = gen_graph_matrix(distance_matrix, assignement_matrix)

for round in range(cfg["comm_round"]):
    party_list_this_round = party_list_rounds[round]
    if args.sample_fraction < 1.0:
        print(f'>> Clients in this round : {party_list_this_round}')
    nets_this_round = {k: local_models[k] for k in party_list_this_round}
    nets_param_start = {k: copy.deepcopy(local_models[k]) for k in party_list_this_round}

    mean_personalized_acc = local_train_pfedgraph(args, round, nets_this_round, cluster_model_vectors, datasets, data_distributions, val, best_val_acc_list, best_test_acc_list, benign_client_list)
   
    total_data_points = sum([datasets[k].x.shape[0] for k in party_list_this_round])

    fed_avg_freqs = {k: datasets[k].x.shape[0] / total_data_points for k in party_list_this_round}

    manipulate_gradient(args, None, nets_this_round, benign_client_list, nets_param_start)
    
    if (not args.ppda) :
        graph_matrix = update_graph_matrix_neighbor(graph_matrix, nets_this_round, global_parameters, dw, fed_avg_freqs, args.alpha, args.difference_measure)   # Graph Matrix is not normalized yet
    
    cluster_model_vectors = aggregation_by_graph(cfg, graph_matrix, nets_this_round, global_parameters)                                                    # Aggregation weight is normalized here

    print('>> (Current) Round {} | Local Per: {:.5f}'.format(round, mean_personalized_acc))
    print('-'*80)
 