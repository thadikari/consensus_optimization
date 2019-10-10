import numpy as np


# plotting
# https://stackoverflow.com/questions/44271504/given-an-adjacency-matrix-how-to-draw-a-graph-with-matplotlib
# https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.convert.from_numpy_matrix.html
# https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file
# https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file


def make_adja_mat(pairs, dim):
    A_ = np.ones([dim, dim])
    for i,j in pairs:
        A_[i,j] = 0
        A_[j,i] = 0
    # assuming no self loops
    for i in range(dim): A_[i,i] = 0
    return A_


# Boyd - Distributed Average Consensus with Time-Varying MetropolisWeights
def _metro(W_, A_, dim, degs):
    for i in range(dim):
        for j in range(i+1,dim):
            if A_[i,j]==1:
                W_[i,j] = 1/(1+max(degs[i],degs[j]))
    W_[:] = W_+W_.T
    for i in range(dim): W_[i,i] = 1-sum(W_[i])


# Dual Averaging for Distributed Optimization - Convergence Analysis and Network Scaling
def _lagra(W_, A_, dim, degs):
    W_[:] = np.eye(dim) - (np.diag(degs) - A_)/(1+max(degs))


def make_doubly_stoch(A_, method):
    assert(len(A_.shape)==2)
    assert(A_.shape[0]==A_.shape[1])
    assert(np.array_equal(A_, A_.T))
    assert(np.array_equal(A_, A_.astype(bool)))

    W_ = np.zeros_like(A_)
    dim = A_.shape[0]
    degs = sum(A_) # degrees
    (_metro if method=='metro' else _lagra)(W_, A_, dim, degs)

    assert(np.allclose(sum(W_), np.ones(dim)))
    assert(np.allclose(sum(W_.T), np.ones(dim)))
    # print(W_)
    # print(np.linalg.matrix_power(W_, 100))
    # exit()
    return W_


def doubly_stoch_from_nc_pairs(pairs, dim, method):
    return make_doubly_stoch(make_adja_mat(pairs, dim), method)
