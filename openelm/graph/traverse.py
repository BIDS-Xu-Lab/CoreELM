import numpy as np
from numpy.random import default_rng

def leaves(adj):
    return np.where(np.diff(adj.indptr) == 0)[0].astype(np.int32)

def _dfs_chains(adj_T, leaf_idx, depth):
    stack = [(leaf_idx, [leaf_idx])]
    while stack:
        node, path = stack.pop()
        if len(path) == depth + 1:
            yield np.array(path[::-1], dtype=np.int32)
            continue
        for parent in adj_T.getrow(node).indices:
            stack.append((parent, path + [parent]))

def _random_walk(adj_T, leaf, depth, rng):
    path = [leaf]
    for _ in range(depth):
        parents = adj_T.getrow(path[-1]).indices
        if parents.size == 0:
            return None
        path.append(int(rng.choice(parents)))
    return np.array(path[::-1], dtype=np.int32)

def branch_iterator(adj, depth=5, max_chains=None, seed=None):
    adj_T = adj.T.tocsr()
    leaf_nodes = leaves(adj)

    if seed is not None:
        rng = default_rng(seed)
        seen = set()
        count = 0
        max_attempts = max(max_chains * 20, 10_000_000) if max_chains else 10_000_000
        attempts = 0
        while (max_chains is None or count < max_chains) and attempts < max_attempts:
            chain = _random_walk(adj_T, int(rng.choice(leaf_nodes)), depth, rng)
            if chain is None:
                attempts += 1
                continue
            key = chain.tobytes()
            if key not in seen:
                seen.add(key)
                yield chain
                count += 1
            attempts += 1
    else:
        count = 0
        for leaf in leaf_nodes:
            for chain in _dfs_chains(adj_T, leaf, depth):
                yield chain
                count += 1
                if max_chains is not None and count >= max_chains:
                    return

def edge_iter(adj):
    cx = adj.tocoo()
    for parent, child in zip(cx.row, cx.col):
        yield np.array([parent, child])
