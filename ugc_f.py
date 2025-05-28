import torch
from ConsistentHashing import *
import UGC_binwidth_finder
import torch.nn.functional as F

coar_device = 'cpu'

def ugc(dataset, coarsening_ratio, i):
    hasher = ConsistentHashing(input_dim=dataset.x.shape[1], proj_dim=1000)
    bin_width, _ = UGC_binwidth_finder.find_Binwidth(dataset, 1-coarsening_ratio)

    Bin_values = hasher.UGC_hashed_values(dataset, function='dot')
    summary_dict = hasher.UGC_partition([bin_width], Bin_values)

    supernode_dict = summary_dict[bin_width]

    rr = 1 - len(supernode_dict.keys())/dataset.x.shape[0]
    reduced_percentage = rr

    print(f'Subgraph-{i} reduced by: {reduced_percentage} percent. \n Now we have {len(supernode_dict.keys())} supernodes; Starting nodes were: {dataset.x.shape[0]}')

    # Send the data back to cpu
    dataset.to(coar_device)

    C = torch.zeros(dataset.x.shape[0] , len(supernode_dict.keys()))
    zero_list = torch.ones(len(supernode_dict.keys()), dtype=torch.bool)

    for super_idx, node_list in enumerate(supernode_dict.values()):
        for node in node_list:
            C[node][super_idx] = 1
            zero_list[super_idx] = zero_list[super_idx] and (dataset.test_mask[node])
    C_diag = torch.sum(C, dim=0)
    P = torch.sparse.mm(C,(torch.diag(torch.pow(C_diag, -1/2))))
    P = P.to_sparse().to_dense()

    return F.normalize(P, p=1.0, dim=1), zero_list