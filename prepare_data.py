from graphdataset import graphdataset_read
import numpy as np
import torch

def get_dataloader(args,cfg):

    #graph datasets
    if args.dataset in ('cora','pubmed', 'citeseer', 'ppi'):
            #PPDA datasets(use old partitions)
            if args.load_graph_path is not None:
                datasets = np.load(args.load_graph_path + 'datasets.pkl', allow_pickle=True)
                traindata_cls_counts_npy = np.load(args.load_graph_path + 'traindata_cls_counts_npy.npy', allow_pickle=True)
                data_distributions = np.load(args.load_graph_path + 'data_distributions.npy', allow_pickle=True)
                val_graph = torch.load(args.load_graph_path + 'val.pt')
                return datasets, traindata_cls_counts_npy, data_distributions, val_graph
            #Create new partitions.
            else:   
                datasets, traindata_cls_counts_npy, data_distributions, val_graph, val_cls_count_npy = graphdataset_read(args.dataset, args.n_parties, args.partition, args.beta, cfg['classes_size'])
                return datasets, traindata_cls_counts_npy, data_distributions    
    