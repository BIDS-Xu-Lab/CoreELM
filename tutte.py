import argparse
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from pathlib import Path
from openelm.config import load_config
from openelm.graph.traverse import leaves

def main():
    parser = argparse.ArgumentParser(description="Tutte embedding of citation graph.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.variant)
    graph_shared       = Path(cfg.paths.graph_shared)
    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    embed_dim          = cfg.embed_abstracts.embed_dim

    print("Loading graph...")
    adj = sp.load_npz(graph_shared / "graph_adj.npz")
    n   = adj.shape[0]

    print("Loading embeddings...")
    emb = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(n, embed_dim)
    )

    # Boundary = leaves (most recent papers, no citers in dataset)
    print("Identifying boundary/interior nodes...")
    boundary = leaves(adj)
    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundary] = True
    interior = np.where(~is_boundary)[0]
    print(f"  {len(boundary):,} boundary  |  {len(interior):,} interior")

    # Centroid of boundary embeddings as affine origin, renormalized
    print("Computing origin...")
    boundary_embs = emb[boundary].astype(np.float64)
    origin = boundary_embs.mean(axis=0)
    origin /= np.linalg.norm(origin)

    # Angular position: cosine distance from origin → angle in [0, 2π]
    cos_dist = 1.0 - (boundary_embs @ origin)   # (B,), approx [0, 1]
    angles   = 2.0 * np.pi * cos_dist
    bx = np.cos(angles)
    by = np.sin(angles)

    # Undirected Laplacian
    print("Building Laplacian...")
    adj_sym = (adj + adj.T).astype(np.float64)
    adj_sym = (adj_sym > 0).astype(np.float64).tocsr()
    deg     = np.array(adj_sym.sum(axis=1)).flatten()
    L       = sp.diags(deg, format="csr") - adj_sym

    # Partition: L_II * x_I = -L_IB * x_B
    L_II = L[interior, :][:, interior]
    L_IB = L[interior, :][:, boundary]

    print("Solving Laplacian system...")
    x_I = spsolve(L_II, -(L_IB @ bx))
    y_I = spsolve(L_II, -(L_IB @ by))

    x = np.zeros(n)
    y = np.zeros(n)
    x[boundary] = bx;  y[boundary] = by
    x[interior] = x_I; y[interior] = y_I

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(x[interior], y[interior], s=0.3, alpha=0.15, c="steelblue",  linewidths=0, label="interior (cited)")
    ax.scatter(x[boundary], y[boundary], s=0.3, alpha=0.15, c="tomato",     linewidths=0, label="boundary (uncited)")
    ax.set_aspect("equal")
    ax.legend(markerscale=15, loc="upper right")
    ax.set_title("Tutte Embedding — Citation Graph")

    out = args.output or str(graph_outputd / "tutte.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()
