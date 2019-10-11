import numpy as np


# not-connected pairs in adjacency matrix
# zero-based indexing from vertices
def make_adja_mat_not_edges(dim, not_edges):
    A_ = np.ones([dim, dim])
    for i,j in not_edges:
        A_[i,j] = 0
        A_[j,i] = 0
    # assuming no self loops
    for i in range(dim): A_[i,i] = 0
    return A_


# connected pairs in adjacency matrix
# zero-based indexing from vertices
def make_adja_mat_edges(dim, edges, bidir=True):
    A_ = np.zeros([dim, dim])
    for i,j in edges:
        A_[i,j] = 1
        if bidir: A_[j,i] = 1
    return A_


graph_defs = {
    'wk_10': make_adja_mat_not_edges(10,
    ((0,1), (0,3), (0,5), (0,9), (1,4), (1,5), (1,8),
     (2,4), (2,7), (2,6), (3,5), (3,6), (3,7),
     (4,6), (4,7), (4,9), (5,8), (6,7), (6,9), (7,8), (8,9))),
    'wk_4': make_adja_mat_edges(4,
    ((0,1),(1,2),(1,3),(3,2)))
}


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


def doubly_stoch_from_not_edges(not_edges, dim, method):
    return make_doubly_stoch(make_adja_mat_not_edges(not_edges, dim), method)


# plotting
# https://stackoverflow.com/questions/44271504/given-an-adjacency-matrix-how-to-draw-a-graph-with-matplotlib
# https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.convert.from_numpy_matrix.html
# https://stackoverflow.com/questions/29572623/plot-networkx-graph-from-adjacency-matrix-in-csv-file
def plot(A_):
    import matplotlib.pyplot as plt
    import networkx as nx

    G_ = nx.from_numpy_matrix(A_)
    nx.draw_spring(G_)
    plt.axis('equal')
    plt.show()


def main():
    plot(default_defs['wk_4'])
    # plot(graph_defs['wk_10'])


if __name__ == '__main__': main()
