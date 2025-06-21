from graphdataset import graphdataset_read
import numpy as np
import torch

def get_dataloader(args,cfg, reqs):

    #graph datasets
    if args.dataset_name in ('cora','pubmed', 'citeseer', 'ppi'):
            #PPDA datasets(use old partitions)
            if args.data_directory is not None:
                datasets = reqs['datasets']
                for dataset in datasets:
                    dataset.train_mask = ~ dataset.test_mask
                # traindata_cls_counts_npy = np.load(args.data_directory + f'traindata_cls_counts_npy_{args.nAnchors}.npy', allow_pickle=True)
                # data_distributions = np.load(args.data_directory + f'data_distributions_{args.nAnchors}.npy', allow_pickle=True)
                # val_graph = torch.load(args.data_directory + f'val_{args.nAnchors}.pt')
                traindata_cls_counts_npy = reqs['traindata_cls_counts_npy']
                data_distributions = reqs['data_distributions']
                val_graph = reqs['val_graph']
                return datasets, traindata_cls_counts_npy, data_distributions, val_graph
            #Create new partitions.
            else:   
                datasets, traindata_cls_counts_npy, data_distributions, val_graph, val_cls_count_npy = graphdataset_read(args.dataset, args.n_parties, args.partition, args.beta, cfg['classes_size'])
                return datasets, traindata_cls_counts_npy, data_distributions    
