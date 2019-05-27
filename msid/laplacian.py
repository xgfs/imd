import numpy as np

from scipy.sparse import lil_matrix, diags, eye


def np_euc_cdist(data):
    dd = np.sum(data*data, axis=1)
    dist = -2*np.dot(data, data.T)
    dist += dd + dd[:, np.newaxis] 
    np.fill_diagonal(dist, 0)
    np.sqrt(dist, dist)
    return dist


def construct_graph_sparse(data, k):
    n = len(data)
    spmat = lil_matrix((n, n))
    dd = np.sum(data*data, axis=1)
    
    for i in range(n):
        dists = dd - 2*data[i, :].dot(data.T)
        inds = np.argpartition(dists, k+1)[:k+1]
        inds = inds[inds!=i]
        spmat[i, inds] = 1
            
    return spmat.tocsr()


def construct_graph_kgraph(data, k):
    import pykgraph

    n = len(data)
    spmat = lil_matrix((n, n))
    index = pykgraph.KGraph(data, 'euclidean')
    index.build(reverse=0, K=2 * k + 1, L=2 * k + 50)
    result = index.search(data, K=k + 1)[:, 1:]
    spmat[np.repeat(np.arange(n), k, 0), result.ravel()] = 1
    return spmat.tocsr()


def _laplacian_sparse(A, normalized=True):
    D = A.sum(1).A1
    if normalized:
        Dsqrt = diags(1/np.sqrt(D))
        L = eye(A.shape[0]) - Dsqrt.dot(A).dot(Dsqrt)
    else:
        L = diags(D) - A
    return L