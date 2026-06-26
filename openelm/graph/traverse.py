import numpy as np
from numpy.random import default_rng

def leaves(adj):
    return np.where(np.diff(adj.indptr) == 0)[0].astype(np.int32)

def _dfs_chains(adj_T, leaf_idx, depth):
    stack = [(leaf_idx, [leaf_idx])]
    while stack:
        node, path = stack.pop()
        if len(path) == depth + 1:
            yield np.array(path[::-1])
            continue
        parents = adj_T.getrow(node).indices
        for parent in parents:
            stack.append((parent, path + [parent]))

def branch_iterator(adj, depth=5):
    adj_T = adj.T.tocsr()
    for leaf in leaves(adj):
        yield from _dfs_chains(adj_T, leaf, depth)

def edge_iter(adj):
    cx = adj.tocoo()
    for parent, child in zip(cx.row, cx.col):
        yield np.array([parent, child])
