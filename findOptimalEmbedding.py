import numpy as np
import torch
import torch.nn as nn
import sys
from torch.autograd import Variable
from math_utils import * # math_utils contains all the math formulas we use

def l2_loss(embedding, graph, dist_function):
    
    # split tensor shape = (num_vertices, dim) into num_vertices number of tensors shape = (dim).
    embedding_tuple = torch.split(embedding, 1)    
    
    # loss function is the sum of l2 norm (no sqrt) between the space distance and tree distance        
    loss = Variable(torch.FloatTensor(torch.zeros(1)))

    # calculate the distance between embedding vectors
    dist_tensor = []
    for i_idx, i in enumerate(embedding_tuple):
        for j_idx, j in enumerate(embedding_tuple):
            if i_idx <= j_idx: # when i_idx==j_idx (dist=0) as it will lead to NaN loss in backprop
                continue
            dist_tensor.append((dist_function(i,j) - graph[i_idx][j_idx]).pow(2))

    # stack the list of calculated distance
    dist_tensor = torch.stack(dist_tensor)

    #loss = (calculated_dist_tensor - graph_variable).pow(2).sum()
    loss = dist_tensor.sum()
    return loss

def findOptimalEmbedding(graph, embedding_dim=2):
    """
    Given a tree (or even graph) with its distance defined in a matrix, 
    find the optimal embedding in Euclidean space and hyperbolic space.
    Inputs:
        graph - a matrix (list or numpy array) with shape = (num_vertices, num_vertices).
                0 on the diagonal and its a symmetric matrix.
    Outputs:
        saves euclidean embedding - shape = (num_vertices, 2). dimension = 2 for visualization.
        saves hyperbolic embedding - shape = (num_vertices, 2). dimension = 2 for visualization.
    """
    
    print('Finding Optimal Embedding')
    
    # hyperparameters
    lr = 1e-3
    num_updates = 5000    
    num_vertices = len(graph)    
    
    # initialize euclidean embedding tensor ~ uniform distribution in range [0, 1)
    euclid_embedding = Variable(torch.rand(num_vertices, embedding_dim).type(torch.FloatTensor), requires_grad=True)
    
    # initialize euclidean embedding tensor ~ uniform distribution in range [0, 0.001)
    hyp_embedding = Variable(torch.div(torch.rand(num_vertices, embedding_dim), 1000).type(torch.FloatTensor), requires_grad=True)
    
    print('start optimizing embedding with lr = %f, total number of updates = %i' %(lr, num_updates))
    for t in range(num_updates):
        
        # l2_loss function is the sum of l2 norm (no sqrt) between the space distance and tree distance        
        euclid_loss = l2_loss(euclid_embedding, graph, euclid_dist)
        hyp_loss = l2_loss(hyp_embedding, graph, hyp_dist)
        
        # print out loss in console
        sys.stdout.write('\r' + ('%i: euclid loss = %f, hyperbolic loss = %f' % (t, euclid_loss.data[0],  hyp_loss.data[0])))
        sys.stdout.flush() 
        
        # using autograd, get gradients for embedding tensors
        euclid_loss.backward()
        hyp_loss.backward()
        
        # Update weights using gradient descent
        euclid_embedding.data -= lr * euclid_embedding.grad.data
        hyp_embedding.data -= lr *inverse_metric_tensor(hyp_embedding)*hyp_embedding.grad.data
        
        # Manually zero the gradients after updating weights
        euclid_embedding.grad.data.zero_()
        hyp_embedding.grad.data.zero_()        
        
    print('finished optimization!')
    np.save('euclid_embedding.npy', euclid_embedding.data.numpy())
    np.save('hyp_embedding.npy', hyp_embedding.data.numpy())
    print('Saved Euclidean embedding to euclidean_embedding.npy and hyperbolic embedding to hyp_embedding.npy !')

if __name__=="__main__":
                           
    #    edge weight matrix
    #    A  B  C  D  E  F  G
    A = [0, 1, 2, 2, 1, 2, 2]
    B = [1, 0, 1, 1, 2, 3, 3]
    C = [2, 1, 0, 2, 3, 4, 4]
    D = [2, 1, 2, 0, 3, 4, 4]
    E = [1, 2, 3, 3, 0, 1, 1]
    F = [2, 3, 4, 4, 1, 0, 2]
    G = [2, 3, 4, 4, 1, 2, 0]
    
    graph = [A, B, C, D, E, F, G]                           
    findOptimalEmbedding(graph)
                           
                           
                           