import numpy as np
import scipy.sparse as sps

from .slq import slq_red_var
from .laplacian import construct_graph_sparse, construct_graph_kgraph, _laplacian_sparse

EPSILON = 1e-6
NORMALIZATION = 1e6


def _build_graph(data, k=5, graph_builder='sparse', normalized=True):
    """
    Return Laplacian from data or load preconstructed from path

    Arguments:
        data: samples
        k: number of neighbours for graph construction
        graph_builder: if 'kgraph', use faster graph construction
        normalized: if True, use nnormalized Laplacian
    Returns:
        L: Laplacian of the graph constructed with data
    """
    if graph_builder == 'sparse':
        A = construct_graph_sparse(data, k)
    elif graph_builder == 'kgraph':
        A = construct_graph_kgraph(data, k)
    else:
        raise Exception('Please specify graph builder: sparse or kgraph.')
    A = (A + A.T) / 2
    A.data = np.ones(A.data.shape)
    L = _laplacian_sparse(A, normalized)
    return L


def _normalize_msid(msid, normalization, n, k, ts):
    normed_msid = msid.copy()
    if normalization == 'empty':
        normed_msid /= n
    elif normalization == 'complete':
        normed_msid /= (1 + (n - 1) * np.exp(-(1 + 1 / (n - 1)) * ts))
    elif normalization == 'er':
        xs = np.linspace(0, 1, n)
        er_spectrum = 4 / np.sqrt(k) * xs + 1 - 2 / np.sqrt(k)
        er_msid = np.exp(-np.outer(ts, er_spectrum)).sum(-1)
        normed_msid = normed_msid / (er_msid + EPSILON)
    elif normalization == 'none' or normalization is None:
        pass
    else:
        raise ValueError('Unknown normalization parameter!')
    return normed_msid


def msid_score(x, y, ts=np.logspace(-1, 1, 256), k=5, m=10, niters=100, rademacher=False, graph_builder='sparse',
              msid_mode='max', normalized_laplacian=True, normalize='empty'):
    '''
    Compute the msid score between two samples, x and y
    Arguments:
        x: x samples
        y: y samples
        ts: temperature values
        k: number of neighbours for graph construction
        m: Lanczos steps in SLQ
        niters: number of starting random vectors for SLQ
        rademacher: if True, sample random vectors from Rademacher distributions, else sample standard normal distribution
        graph_builder: if 'kgraph', uses faster graph construction (options: 'sparse', 'kgraph')
        msid_mode: 'l2' to compute the l2 norm of the distance between `msid1` and `msid2`;
                'max' to find the maximum abosulute difference between two descriptors over temperature
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi
                normalization, 'none' for no normalization
    Returns:
        msid_score: the scalar value of the distance between discriptors
    '''
    normed_msidx = msid_descriptor(x, ts, k, m, niters, rademacher, graph_builder, normalized_laplacian, normalize)
    normed_msidy = msid_descriptor(y, ts, k, m, niters, rademacher, graph_builder, normalized_laplacian, normalize)

    c = np.exp(-2 * (ts + 1 / ts))

    if msid_mode == 'l2':
        score = np.linalg.norm(normed_msidx - normed_msidy)
    elif msid_mode == 'max':
        score = np.amax(c * np.abs(normed_msidx - normed_msidy))
    else:
        raise Exception('Use either l2 or max mode.')

    return score


def msid_descriptor(x, ts=np.logspace(-1, 1, 256), k=5, m=10, niters=100, rademacher=False, graph_builder='sparse',
              normalized_laplacian=True, normalize='empty'):
    '''
    Compute the msid descriptor for a single sample x
    Arguments:
        x: x samples
        ts: temperature values
        k: number of neighbours for graph construction
        m: Lanczos steps in SLQ
        niters: number of starting random vectors for SLQ
        rademacher: if True, sample random vectors from Rademacher distributions, else sample standard normal distribution
        graph_builder: if 'kgraph', uses faster graph construction (options: 'sparse', 'kgraph')
        normalized_laplacian: if True, use normalized Laplacian
        normalize: 'empty' for average heat kernel (corresponds to the empty graph normalization of NetLSD),
                'complete' for the complete, 'er' for erdos-renyi
                normalization, 'none' for no normalization
    Returns:
        normed_msidx: normalized msid descriptor
    '''
    Lx = _build_graph(x, k, graph_builder, normalized_laplacian)

    nx = Lx.shape[0]
    msidx = slq_red_var(Lx, m, niters, ts, rademacher)

    normed_msidx = _normalize_msid(msidx, normalize, nx, k, ts) * NORMALIZATION

    return normed_msidx
