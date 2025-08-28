import argparse
import random
import time
import warnings
from math import log2
import pycuda.driver as cuda
import os
from ugc_f import ugc
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, dense_to_sparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import pylab
import scipy
import torch
import torch_geometric
import cupy as cp
import cupyx.scipy.spatial.distance as test

from scipy import spatial, sparse as sp
from scipy.linalg import fractional_matrix_power
from scipy.spatial.distance import cdist, pdist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from tqdm import tqdm

def update_table(dataset, accuracy, anchors, coarsening_ratio, ifppda=True, path="cora_exp.csv"):
    row = {"dataset": dataset, '#Anchors': anchors, "accuracy": accuracy, 'PPDA': ifppda, 'r': coarsening_ratio}

    # Check if file exists and is not empty
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    
    df.to_csv(path, index=False)
    print(f"Results saved: {accuracy} acc on {dataset} with {anchors} anchors and coarsening ratio {coarsening_ratio}")

def Hbeta(D=np.array([]), beta=1.0):
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    # print("distamce matrix", D)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    print("P complete", P, P.shape)
    return P

def tsne(X=np.array([]), no_dims=2, perplexity=30.0, random_state=None,max_iter=1000, initial_momentum=0.4, final_momentum=0.8, eta=100, min_gain=0.01, early_exag=2):

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    (n_tsne, d) = X.shape

    Y = np.random.randn(n_tsne, no_dims)
    dY = np.zeros((n_tsne, no_dims))
    iY = np.zeros((n_tsne, no_dims))
    gains = np.ones((n_tsne, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * early_exag								# early exaggeration
    P = np.maximum(P, 1e-12)


    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n_tsne):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n_tsne, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / early_exag

    # Return solution
    return Y


def dist_approx(*client_data, X_a, n):
    num_clients = len(client_data)  # Number of clients
    print("num_clients", num_clients)
    n_clients = [data.shape[0] for data in client_data]  # Number of samples in each client's data
    print("n_clients", n_clients)

    DA = cdist(X_a, X_a, metric='euclidean')  # Anchor to anchor distance
    D = np.zeros((sum(n_clients) + X_a.shape[0], sum(n_clients) + X_a.shape[0]))  # Initialize distance matrix

    print('hi')
    # Compute distances between non-anchor data of clients
    for i, C_na_data in enumerate(client_data):
        for j, C_na_data_other in enumerate(client_data):
            if i == j:
                D_offset = sum(n_clients[:i])
                D[D_offset:D_offset + n_clients[i], D_offset:D_offset + n_clients[i]] = cdist(C_na_data, C_na_data, metric='euclidean')
    # Compute distances from each client to anchors
    DNA = [cdist(C_na_data, X_a, metric='euclidean') for C_na_data in client_data]

    for i in range(num_clients):
        D_offset = sum(n_clients[:i])
        D[D_offset:D_offset + n_clients[i], -X_a.shape[0]:] = DNA[i]
        D[-X_a.shape[0]:, D_offset:D_offset + n_clients[i]] = DNA[i].T
    # Block construction for the full distance matrix
    D[-X_a.shape[0]:, -X_a.shape[0]:] = DA  # Anchor distances to anchors
    D = (D + D.T) / 2
    W_1 = np.zeros(D.shape)
    start_idx = 0
    for i in range(num_clients):
        end_idx = start_idx + n_clients[i]
        W_1[start_idx:end_idx, start_idx:end_idx] = 1
        W_1[start_idx:end_idx, -X_a.shape[0]:] = 1
        W_1[-X_a.shape[0]:, start_idx:end_idx] = 1
        start_idx = end_idx
    W_1[-X_a.shape[0]:, -X_a.shape[0]:] = 1
    # print("W_1 Matrix:")

    V = np.array(np.diag(np.matmul(W_1, np.ones(W_1.shape[0]))) - W_1)
    V1 = V[:sum(n_clients), :sum(n_clients)]
    V2 = V[:sum(n_clients), sum(n_clients):]

    return D, V, V1, V2, W_1, DA, *DNA

def mat(D , V1 , V2 , W_1 , DA , X_a , n_list , n ,d , Zu_samples):
    Zu = []
    start_idx = 0
    for n_i in n_list:
        Zu_i = Zu_samples[start_idx:start_idx + n_i, :]
        Zu.append(Zu_i)
        start_idx += n_i

    #print(len(X_a[0]))
    #print(Zu)
    #Zu = Zu_samples[:n,:]


    DNA_new = []
    #DNA_new = cdist(Zu , X_a , metric = 'euclidean')
    #X_a = tsne(X_a)
    for i in range(len(n_list)):
        client_distances = cdist(Zu[i], X_a, metric='euclidean')
        DNA_new.append(client_distances)

    epsilon = 1e-3
    epochs = 2000
    loss = []
    V1_cp = cp.asarray(V1)
    V1_inv = cp.linalg.pinv(V1_cp)
    V1_inv =  cp.asnumpy(V1_inv)
    W_new = np.multiply(W_1, D)

    Dnew_list = [cdist(Zu[i], Zu[i], metric='euclidean') for i in range(len(n_list))]

    zero_blocks = []
    for i in range(len(n_list)):
        for j in range(len(n_list)):
            if i != j:
                zero_blocks.append(np.zeros((n_list[i], n_list[j])))  # Shape (n_i, n_j)

    D_new = []
    for i in range(len(n_list)):
        row = []
        for j in range(len(n_list)):
            if i == j:
                row.append(Dnew_list[i])
            else:
                row.append(zero_blocks.pop(0))
        row.append(DNA_new[i])
        D_new.append(row)

    anchor_row = []
    for i in range(len(n_list)):
        anchor_row.append(DNA_new[i].T)
    anchor_row.append(DA)

    D_new.append(anchor_row)

    try:
        D_new = np.block(D_new)
        #print("\nBlock matrix created successfully.")
    except ValueError as e:
        print("\nError creating block matrix:", e)

    return D_new

def MDS_X(D, V1, V2, W_1, DA, X_a, n_list, n, d):
    D_inv = np.reciprocal(D, out=np.zeros_like(D), where=(D != 0))
    L_D_inv = np.diag(np.matmul(D_inv, np.ones(D_inv.shape[0]))) - D_inv

    np.random.seed(50)
    L_D_inv_cp =  cp.asarray(L_D_inv)
    print(f'Inverse of L_D to be taken of size: {L_D_inv_cp.shape}')
    temp = cp.linalg.pinv(L_D_inv_cp)
    temp =  cp.asnumpy(temp)
    Zu_samples = np.random.multivariate_normal(np.zeros(n + X_a.shape[0]), temp, d).T
    del temp, L_D_inv_cp
    Zu = []
    start_idx = 0
    for n_i in n_list:
        Zu_i = Zu_samples[start_idx:start_idx + n_i, :]
        Zu.append(Zu_i)
        start_idx += n_i

    DNA_new = []
    for i in range(len(n_list)):
        client_distances = cdist(Zu[i], X_a, metric='euclidean')
        DNA_new.append(client_distances)

    epsilon = 1e-3
    epochs = 1000
    loss = []
    V1_cp = cp.asarray(V1)
    print(f"Taking inverse of V of size: {V1_cp.shape}")
    V1_inv = cp.linalg.pinv(V1_cp)
    V1_inv =  cp.asnumpy(V1_inv)
    W_new = np.multiply(W_1, D)

    Dnew_list = [cdist(Zu[i], Zu[i], metric='euclidean') for i in range(len(n_list))]

    zero_blocks = []
    for i in range(len(n_list)):
        for j in range(len(n_list)):
            if i != j:
                zero_blocks.append(np.zeros((n_list[i], n_list[j])))  # Shape (n_i, n_j)

    D_new = []
    for i in range(len(n_list)):
        row = []
        for j in range(len(n_list)):
            if i == j:
                row.append(Dnew_list[i])
            else:
                row.append(zero_blocks.pop(0))
        row.append(DNA_new[i])
        D_new.append(row)

    anchor_row = []
    for i in range(len(n_list)):
        anchor_row.append(DNA_new[i].T)
    anchor_row.append(DA)

    D_new.append(anchor_row)

    try:
        D_new = np.block(D_new)
        print("\nBlock matrix created successfully.")
    except ValueError as e:
        print("\nError creating block matrix:", e)

    Zu_combined = np.vstack(Zu)
    n_total = sum(n_list)
    n_anchors = X_a.shape[0]

    for t in tqdm(range(epochs)):
        
        W_final = np.divide(W_new, D_new, out=np.zeros_like(W_new), where=(D_new != 0))
        B_Z = np.diag(np.matmul(W_final, np.ones(W_final.shape[0]))) - W_final
        BZ1 = B_Z[:n_total, :n_total]
        BZ2 = B_Z[:n_total, n_total:n_total + n_anchors]
        term1 = np.matmul(BZ1, Zu_combined)
        term2_temp = BZ2 - V2
        term2 = np.matmul(term2_temp, X_a)
        X_final = np.matmul(V1_inv, term1 + term2)
        # DNA_new = cdist(X_final, X_a, metric='euclidean')
        #D_new = np.array(np.vstack((np.hstack((np.zeros((X_final.shape[0], X_final.shape[0])), DNA_new)), np.hstack((DNA_new.T, DA)))))
        #profiler = LineProfiler()
        #profiler.add_function(mat)
        #profiler.enable()
        D_new = mat(D , V1 , V2 , W_1 , DA , X_a , n_list , n ,d , X_final)
        #profiler.disable()
        #profiler.print_stats()
        D_inv_new = np.reciprocal(D_new, out=np.zeros_like(D_new), where=(D_new != 0))
        W_upper_triag = np.array(D_inv_new[np.triu_indices(D_inv.shape[0], k=1)])
        C = np.square(D - D_new)
        D_upper_triag = C[np.triu_indices(C.shape[0], k=1)]
        stress = np.dot(W_upper_triag, D_upper_triag)
        loss.append(stress)
        Zu_combined = X_final

        if t % 10 == 0:
            print(stress)
        if t!=0:
            if abs(loss[t]-loss[t-1]) < epsilon:
                break

    return X_final, loss


def dist_error(X_na, X_final):

    D_true = cdist(X_na, X_na, metric='euclidean')
    z_true = spatial.distance.squareform(D_true)

    D_esti = cdist(X_final, X_final, metric='euclidean')
    z_esti = spatial.distance.squareform(D_esti)
    Error = np.linalg.norm((D_true - D_esti), 'fro')/ np.linalg.norm((D_true), 'fro')

    return Error, D_true, D_esti, z_true, z_esti


def check_score(D_true, D_approx,k):
      f_scores = []
      for i in range(D_true.shape[0]):
        list1 =  np.argsort(D_true[i])
        list2 =  np.argsort(D_approx[i])
        newlist1 = list1[1:k]
        newlist2 = list2[1:k]
        count = 0
        for p in range(k-1):
          for q in range(k-1):
            if(newlist1[p] == newlist2[q]):
              count += 1
              break

        f_score = 2*count/(2*count + (k- 1 - count))
        f_scores.append(f_score)
      avg_f_score = sum(f_scores)/len(f_scores)

      return avg_f_score

def run_ppda(client_features, X_a, X_na, output_directory):

    client_data = [client_features[i].astype('float') for i in range(len(client_features))]
    n_sizes = [client_data[i].shape[0] for i in range(len(client_features))]
    D, V, V1, V2, W_1, DA, *DNA = dist_approx(*client_data, X_a=X_a.astype('float'), n=X_a.shape[0])
    X_a=X_a.astype('float')
    print('dist_approx done')
    
    # Perform MDS
    X_final, loss = MDS_X(D, V1, V2, W_1, DA, X_a, n_sizes, X_a.shape[0], X_a.shape[1])
    os.makedirs(output_directory, exist_ok=True)

    # np.save(output_directory + f"X_final_{dataset_name}_{args.nAnchors}.npy", X_final)
    print("Calculating the distance error between X_na and X_final of shapes", X_na.shape, X_final.shape, "X_na looks like", X_na, "X_final looks like", X_final)
    error, D_true, D_esti, z_true, z_esti = dist_error(X_na.astype('float'), X_final)
    # np.save(output_directory + f"D_esti_{args.nAnchors}" + dataset_name + "_.npy", D_esti)
    print("Error in distance approximation: ", error)
    fscore = check_score(D_true, D_esti, 11)
    print("F-score: ", fscore)

    #pylab.scatter(X_final[:, 0], X_final[:, 1], 20, labels)
    #pylab.savefig(output_directory + "vis.png")  # Save to the output filename
    
    return D_esti, X_final, loss