import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def clustering(data, no_of_hash, bin_width): #no_of_hash= number_of_projectors(1000), bin_width=bw line: 18
  Wl = (torch.randn(no_of_hash, data.x.shape[1]) * (1 / np.sqrt(no_of_hash))).to(device=device)
  data = data.to(device=device)
  Bin_values = torch.matmul(data.x, Wl.T) #got bin_values

  temp = torch.floor((1/bin_width)*(Bin_values))  

  cluster, _ = torch.mode(temp, dim = 1)
  dict_hash_indices = {}
  no_nodes = Bin_values.shape[0]
  for i in range(no_nodes):
      dict_hash_indices[i] = int(cluster[i]) #.to('cpu')

  del Wl

  return dict_hash_indices

def clustering_WWL(data, no_of_hash, bin_width): #no_of_hash= number_of_projectors(1000), bin_width=bw line: 18
  data = data.to(device=device)
  Wl = (torch.randn(no_of_hash, 2*data.x.shape[1]) * (1 / np.sqrt(no_of_hash))).to(device=device)

  g_adj = to_dense_adj(data.edge_index, edge_attr= data.edge_attr)[0]
  wlf = (1/2)*torch.matmul(g_adj,data.x) + data.x
  augdata = torch.cat((data.x, wlf), dim = 1)  
    
  Bin_values = torch.matmul(augdata, Wl.T).to(device=device) #got bin_values
  
  temp = torch.floor((1/bin_width)*(Bin_values))  
  
  cluster, _ = torch.mode(temp, dim = 1)
  dict_hash_indices = {}
  no_nodes = Bin_values.shape[0]
  for i in range(no_nodes):
      dict_hash_indices[i] = int(cluster[i]) #.to('cpu')

  del Wl

  return dict_hash_indices

def find_Binwidth(data,coarsening_ratio, precision = 0.005, number_of_projectors=1000):
  bw = 1
  ratio = 1
  counter = 0
#   number_of_projectors = int(np.ceil(np.log(data.x.shape[0])))
  while(abs(ratio - coarsening_ratio) > precision):
    counter = counter + 1
    if(ratio > coarsening_ratio):
      bw = bw*0.5
    else:
      bw = bw*1.5

    g_coarsened = clustering(data,number_of_projectors, bw) #function calling clustering, line: 4
    values = g_coarsened.values() 
    unique_values = set(g_coarsened.values())
    ratio = (1 - (len(unique_values)/len(values)))
  
#   print(ratio, coarsening_ratio, bw)
#   print(counter)

  return bw, ratio


def find_Binwidth_WWL(data,coarsening_ratio, precision = 0.05, number_of_projectors=1000):
  print("this should not be coming here in bindwidth finder")
  exit(1)
  bw = 1
  ratio = 1
  counter = 0
#   number_of_projectors = int(np.ceil(np.log(data.x.shape[0])))
  while(abs(ratio - coarsening_ratio) > precision):
    counter = counter + 1
    if(ratio > coarsening_ratio):
      bw = bw*0.5
    else:
      bw = bw*1.5

    g_coarsened = clustering_WWL(data,number_of_projectors, bw) #function calling clustering, line: 4
    values = g_coarsened.values() 
    unique_values = set(g_coarsened.values())
    ratio = (1 - (len(unique_values)/len(values)))
  
#   print(ratio, coarsening_ratio, bw)
#   print(counter)

  return bw, ratio