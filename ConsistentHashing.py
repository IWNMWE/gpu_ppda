import torch
import torch.nn.functional as F
import numpy as np
import random
from torch_geometric.utils import to_dense_adj

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class ConsistentHashing(torch.nn.Module):
    def __init__(self, input_dim, proj_dim, learnable=False):
        super().__init__()
        self.proj_dim = proj_dim
        if learnable:
            self.W = torch.nn.Parameter(torch.randn(proj_dim, input_dim) * (1 / np.sqrt(proj_dim)))
        else:
            # W = torch.randn(proj_dim, input_dim)
            W = (torch.randn(proj_dim, input_dim) * (1 / np.sqrt(proj_dim))).to(device=device)
            self.register_buffer('W', W)

    def forward(self, data):
        # """
        # 
        # """
        # JL projection : Project node features using JL projection and assign them to supernodes 
        #                 based on average projection values (like placing them on a hash ring).
        

        ## normal projection
        # X_proj = F.linear(data.x, self.W)  # [N, d]
        #########
        
        ####### WWL augmented features
        data = data.to(device)
        # g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
        # wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
        # augdata = torch.cat((data.x, wlf), dim = 1)  
        # # Wl = torch.FloatTensor(self.proj_dim, 2*feature_size).uniform_(0,1)
        
        # X_proj = torch.matmul(augdata, self.W.T)
        ###########################
        
        X_proj = torch.matmul(data.x, self.W.T)

        # node_line_values = F.normalize(X_proj, dim=1)  # [N, d], unit vectors
        node_line_values = X_proj.mean(dim=1)

        
        # Get unique ring positions and their inverse indices
        unique_pos, inverse_indices = torch.unique(node_line_values, return_inverse=True, sorted=True)
        p = unique_pos.size(0)

        # Initialize supernode dictionary: key = supernode ID, value = list of node indices
        supernode_dict = {i: [] for i in range(p)}
        for node_idx, supernode_idx in enumerate(inverse_indices):
            supernode_dict[int(supernode_idx.item())].append(node_idx)

        return supernode_dict
    
    def UGC_partition(self, list_bin_width,Bin_values):
        summary_dict = {}
        for bin_width in list_bin_width:
            # bias = torch.tensor([random.uniform(-bin_width, bin_width) for i in range(self.proj_dim)])
            # temp = torch.floor((1/bin_width)*(Bin_values + bias))

            temp = torch.floor((1/bin_width)*(Bin_values))

            cluster, _ = torch.mode(temp, dim = 1)
            dict_hash_indices = {}
            no_nodes = Bin_values.shape[0]

            for i in range(no_nodes):
                supernode_id = int(cluster[i])
                if supernode_id not in dict_hash_indices:
                    dict_hash_indices[supernode_id] = []
                dict_hash_indices[supernode_id].append(i)

            # for i in range(no_nodes):
            #     dict_hash_indices[i] = int(cluster[i])

            summary_dict[bin_width] = dict_hash_indices 

        return summary_dict
    
    def UGC_hashed_values(self, data, function='dot'):

        if function == 'L2-norm':
            Bin_values = torch.cdist(data.x, self.W, p = 2)
        elif function == 'L1-norm':
            Bin_values = torch.cdist(data.x, self.W, p = 1)
        else:
            #dot
            Bin_values = torch.matmul(data.x, self.W.T)
        
        return Bin_values
    
    def UGC_hashed_values_with_WWL(self, data, function='dot'):

        g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
        wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
        augdata = torch.cat((data.x, wlf), dim = 1)  
        # Wl = torch.FloatTensor(self.proj_dim, 2*feature_size).uniform_(0,1)

        if function == 'L2-norm':
            Bin_values = torch.cdist(augdata, self.W, p = 2)
        elif function == 'L1-norm':
            Bin_values = torch.cdist(augdata, self.W, p = 1)
        else:
            #dot
            Bin_values = torch.matmul(augdata, self.W.T)
        
        return Bin_values
    
        
    def rank_and_sort_supernodes(self, supernode_dict, features):
        mean_values_dict = {}
        for sid, nodes in supernode_dict.items():
            if nodes: 
                mean_val = features[nodes].mean()
            else:
                mean_val = -np.inf 
            mean_values_dict[sid] = mean_val

        ranked = sorted(mean_values_dict.items(), key=lambda x: -x[1])
        
        sorted_supernode_dict = {sid: supernode_dict[sid] for sid, _ in ranked}
        sid_rank_map = {sid: rank for rank, (sid, _) in enumerate(ranked)}
        ordered_mean_values = [mean_values_dict[sid] for sid, _ in ranked]

        return sorted_supernode_dict, sid_rank_map, ordered_mean_values


    
    def coarsen_ring_parallel_gpu(self, supernode_dict, data, num_supernodes):
        """
        Reduce entries on line by merging adjacent positions.
        """
        # Iteratively merge nodes until desired number of supernodes is reached
        keys_now = list(supernode_dict.keys())
        p = len(keys_now)
        num_nodes = data.x.shape[0]
        print("n and desired supernodes ", p, num_supernodes)
        while p > num_supernodes:
            # if p <= 1:
            #     break  # Nothing to merge
            # Select one random position from 0 to p - 2 (so next exists)
            idx = random.randint(0, p - 2)
            # # print(keys_now, idx)
            # k = keys_now[idx]
            # z = keys_now[idx + 1]

            # Merge supernode_dict[z] into supernode_dict[k]
            supernode_dict[keys_now[idx]].extend(supernode_dict[keys_now[idx + 1]])
            del supernode_dict[keys_now[idx + 1]]
            keys_now.remove(keys_now[idx + 1])
            p = p - 1
        
        # Initialize final coarsening matrix (n x num_supernodes)
        print("we already have the coarsened list making C matrix now")
        C = torch.zeros(num_nodes, num_supernodes)
        zero_list = torch.ones(num_supernodes, dtype=torch.bool)

        for super_idx, node_list in enumerate(supernode_dict.values()):
            for node in node_list:
                C[node][super_idx] = 1
                if hasattr(data, 'train_mask'):
                    zero_list[super_idx] = zero_list[super_idx] and (not (data.train_mask)[node])
                else: zero_list[super_idx] = False
                

        # C = []
        return C, supernode_dict, zero_list



