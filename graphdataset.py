import torch.utils.data as data
import numpy as np
from partition_graph import partition_graph
import torch
import os
import glob
from graphdataset import graphdataset_read
from scipy.spatial.distance import cdist
import pickle
import scipy
    

def graphdataset_read(dataset, n_parties, partition, beta, classes, anchors=100):
    if dataset in ("cora", "pubmed", "citeseer"):
        datasets, val_graph = partition_graph(dataset, partition, n_parties, "average", beta, anchors)
        traindata_cls_counts_dict = {}
        traindata_cls_counts_npy = np.array([])
        num_classes = classes
        for data in datasets:
            y_train = data.y[data.test_mask == 0].numpy()
            unq, unq_cnt = np.unique(y_train, return_counts=True)
            for i in range(len(unq)):
                traindata_cls_counts_dict[unq[i]] = unq_cnt[i]
            
            tmp_npy = np.zeros(classes)
            for id, cnt in traindata_cls_counts_dict.items():
                tmp_npy[id] = cnt
            traindata_cls_counts_npy = np.concatenate((traindata_cls_counts_npy, tmp_npy), axis=0)
        traindata_cls_counts_npy = np.reshape(traindata_cls_counts_npy, (-1,num_classes))
        data_distributions = traindata_cls_counts_npy / traindata_cls_counts_npy.sum(axis=1)[:,np.newaxis]

        unq, unq_cnt = np.unique(val_graph.y.numpy(), return_counts=True)
        val_cls_count_npy = np.zeros(classes)
        for i in range(len(unq)):
            val_cls_count_npy[unq[i]] = unq_cnt[i]
        #print(traindata_cls_counts_npy.astype(int))
        #print(val_cls_count_npy.astype(int))
        return datasets, traindata_cls_counts_npy, data_distributions, val_graph, val_cls_count_npy
    elif dataset=='niid':
        if n_parties != 3:
            raise ValueError('NIID dataset only supports 3 parties')
        datasets = []
        num_classes = 2
        traindata_cls_counts_dict = {}
        traindata_cls_counts_npy = np.array([])
        for np_name in glob.glob('/home/sattu/pFedGraph/niid/*.pt'):
            print(np_name)
            data = torch.load(np_name)
            datasets.append(data)

        for data in datasets:
            y_train = data.y.numpy()
            unq, unq_cnt = np.unique(y_train, return_counts=True)
            for i in range(len(unq)):
                traindata_cls_counts_dict[i] = unq_cnt[i]
            tmp_npy = np.zeros(2)
            for id, cnt in traindata_cls_counts_dict.items():
                tmp_npy[id] = cnt
            print(tmp_npy)
            traindata_cls_counts_npy = np.concatenate((traindata_cls_counts_npy, tmp_npy), axis=0)
        print(traindata_cls_counts_npy)
        traindata_cls_counts_npy = np.reshape(traindata_cls_counts_npy, (-1,num_classes))
        data_distributions = traindata_cls_counts_npy / traindata_cls_counts_npy.sum(axis=1)[:,np.newaxis]
        print(traindata_cls_counts_npy.astype(int))
        return datasets, traindata_cls_counts_npy, data_distributions
    elif dataset=='ppi':
        datasets = partition_graph(dataset, partition, n_parties, "average", beta)
        traindata_cls_counts_dict = {}
        traindata_cls_counts_npy = np.array([])
        num_classes = 7
        for data in datasets:
            y_train = data.y[data.test_mask == 0].numpy()
            unq, unq_cnt = np.unique(y_train, return_counts=True)
            for i in range(len(unq)):
                traindata_cls_counts_dict[i] = unq_cnt[i]
            
            tmp_npy = np.zeros(7)
            for id, cnt in traindata_cls_counts_dict.items():
                tmp_npy[id] = cnt
            traindata_cls_counts_npy = np.concatenate((traindata_cls_counts_npy, tmp_npy), axis=0)
        traindata_cls_counts_npy = np.reshape(traindata_cls_counts_npy, (-1,num_classes))
        data_distributions = traindata_cls_counts_npy / traindata_cls_counts_npy.sum(axis=1)[:,np.newaxis]
        print(traindata_cls_counts_npy.astype(int))
        return datasets, traindata_cls_counts_npy, data_distributions
    

def store_partition_graph(dataset_name, beta, distribution, num_clients, base_path, num_anchors=100):
    if dataset_name == 'cora':
        classes = 7
    elif dataset_name == 'citeseer':
        classes = 6
    elif dataset_name == 'pubmed':
        classes = 3
    datasets, traindata_cls_counts_npy, data_distributions, val, _ = graphdataset_read(dataset_name, num_clients, distribution, beta, classes, num_anchors)
    dists = []
    for dataset in datasets:
        dist = cdist(dataset.x, dataset.x)
        dists.append(dist)
    D_partial = scipy.linalg.block_diag(*dists)
    
    name  = base_path + dataset_name + '/' + distribution  + '/' + str(beta).replace('.', '_') 
    os.makedirs(name, exist_ok=True)
    np.save(name + '/D_partial.npy', D_partial)

    with open(name + '/datasets.pkl', "wb") as f:
        pickle.dump(datasets,f)
    np.save(name + '/traindata_cls_counts_npy.npy', traindata_cls_counts_npy)
    np.save(name + '/data_distributions.npy', data_distributions)
    torch.save(val, name + '/val.pt')

    n = 0
    l = [0]
    for dataset in datasets:
        n += dataset.x.shape[0]
        l.append(n)
    C = np.zeros((n, len(datasets)))
    for i in range(len(l) - 1):
        C[l[i]:l[i+1], i] = 1
    
    np.save(name + "/C.npy", C)

    