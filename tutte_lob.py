import argparse
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cspla
import matplotlib.pyplot as plt
from pathlib import Path
from openelm.config import load_config
from openelm.graph.traverse import leaves


def main():
    parser = argparse.ArgumentParser(description="Tutte embedding of citation graph (leaves on boundary).")
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

    print("Identifying boundary/interior nodes...")
    boundary = leaves(adj)
    is_boundary = np.zeros(n, dtype=bool)
    is_boundary[boundary] = True
    interior = np.where(~is_boundary)[0]
    print(f"  {len(boundary):,} boundary (leaves, in-degree=0)  |  {len(interior):,} interior")

    # Boundary positions: cosine distance from centroid → angle in [0, 2π]
    print("Computing boundary positions...")
    boundary_embs = cp.array(emb[boundary].astype(np.float64))
    origin = boundary_embs.mean(axis=0)
    origin /= cp.linalg.norm(origin)
    cos_dist = 1.0 - (boundary_embs @ origin)
    # Normalize to [0, 2π] so boundary spans the full circle
    cos_dist = (cos_dist - cos_dist.min()) / (cos_dist.max() - cos_dist.min())
    angles = 2.0 * np.pi * cos_dist
    bx = cp.cos(angles)
    by = cp.sin(angles)

    # Build undirected Laplacian on CPU, send submatrices to GPU
    print("Building Laplacian...")
    adj_sym = (adj + adj.T).astype(np.float64)
    adj_sym = (adj_sym > 0).astype(np.float64).tocsr()
    deg = np.array(adj_sym.sum(axis=1)).flatten()
    L   = sp.diags(deg, format="csr") - adj_sym

    L_II = csp.csr_matrix(L[interior, :][:, interior])
    L_IB = csp.csr_matrix(L[interior, :][:, boundary])

    print("Solving Laplacian system (GPU CG)...")
    rhs_x = -(L_IB @ bx)
    rhs_y = -(L_IB @ by)
    x_I, _ = cspla.cg(L_II, rhs_x, rtol=1e-6)
    y_I, _ = cspla.cg(L_II, rhs_y, rtol=1e-6)

    x = np.zeros(n);  y = np.zeros(n)
    x[boundary] = cp.asnumpy(bx);  y[boundary] = cp.asnumpy(by)
    x[interior] = cp.asnumpy(x_I); y[interior] = cp.asnumpy(y_I)

    # Log-radial transform for visualization only (raw x/y preserved for DMD)
    r     = np.sqrt(x**2 + y**2)
    r_log = np.log1p(r)
    xv = np.where(r > 0, x / r * r_log, 0.0)
    yv = np.where(r > 0, y / r * r_log, 0.0)

    print("Plotting...")
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(xv[interior], yv[interior], s=0.3, alpha=0.15, c="steelblue", linewidths=0, label="interior (cited)")
    ax.scatter(xv[boundary], yv[boundary], s=0.3, alpha=0.15, c="tomato",    linewidths=0, label="boundary (leaves, uncited)")
    ax.set_aspect("equal")
    ax.legend(markerscale=15, loc="upper right")
    ax.set_title("Tutte Embedding — Citation Graph, Leaves on Boundary (log-radial)")

    out = args.output or str(graph_outputd / "tutte_lob.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
