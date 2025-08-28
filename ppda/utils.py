import numpy as np
import torch

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
    print(assignment_matrix.shape, L.shape)
    L_graph = assignment_matrix.T @ L @ assignment_matrix
    D_graph = np.diag(np.diag(L_graph))
    A_graph = D_graph - L_graph
    A_graph = np.clip(A_graph, 0, None)
    A_sum = np.sum(A_graph, axis=1) 
    A_graph_norm  = A_graph / A_sum[:, np.newaxis]

    return torch.tensor(A_graph_norm)

def prep_ppda(client_pyg_datasets, anchor_datapoints):

    # Prepare data for running ppda.
    X_a = anchor_datapoints.x.numpy().astype('float')
    client_features = [dataset.x.numpy().astype('float') for dataset in client_pyg_datasets]
    X_na = np.concatenate(client_features, axis=0)

    return client_features, X_a, X_na