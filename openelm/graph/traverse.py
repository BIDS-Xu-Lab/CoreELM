## Library Imports
import numpy as np 
import scipy.sparse as sp 
from numpy.random import default_rng

def leaves(adj):
    '''
    using the out-degree measurement from the CSR, return leaf nodes
    '''
    return np.where(np.diff(adj.indptr) == 0)[0].astype(np.int32)
    
def walk_from_leaf(adj, leaf_idx, depth, rng=None):
    path = [leaf_idx]
    lvl = depth
    while lvl != 0:
        parents = adj.T.getrow(path[-1]).indices
        if parents.size > 0:
            path.append(rng.choice(parents) if rng else parents[0])
        else:
            return
        lvl-=1
    return np.array(path[::-1])

def branch_iterator(adj, depth=5, seed=None):
    rng = default_rng(seed) if seed is not None else None
    leaf_nodes = leaves(adj)
    for leaf in leaf_nodes:
        chain = walk_from_leaf(adj, leaf, depth, rng)
        if chain is not None:
            yield chain
