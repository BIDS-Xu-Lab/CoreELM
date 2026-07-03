import argparse
import numpy as np
import scipy.sparse as sp
from pathlib import Path
from openelm.config import load_config
from .build import load_pmids, load_abstracts, fetch_citations, build_edges, build_csr
from .traverse import branch_iterator
from .chains import one_text_chain

def main():
    parser = argparse.ArgumentParser(description="Build Citation DAG from iCite data.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config, args.variant, args.experiment)
    gcfg = cfg.graph_build
    output_dir = Path(cfg.paths.graph_outputd)
    shared_dir = Path(cfg.paths.graph_shared)
    output_dir.mkdir(parents=True, exist_ok=True)
    shared_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PMIDs...")
    pmids, pmid_idx = load_pmids(gcfg.pmidf)
    print(f"  {len(pmids)} PMIDs loaded")

    print("Loading abstracts...")
    abstracts = load_abstracts(gcfg.txt, pmid_idx, keep_labels=gcfg.get('keep_labels', False))
    print(f"  {len(abstracts)} abstracts loaded")

    print("Fetching citations...")
    pmid_tbl = fetch_citations(gcfg.db, pmid_idx)
    print(f"  {len(pmid_tbl)} rows returned")

    print("Building edges...")
    edges = build_edges(pmid_tbl, pmid_idx)
    print(f"  {len(edges)} edges found")

    print("Building CSR...")
    adj = build_csr(edges, len(pmids))

    print("Extracting sample chains...")
    for i, chain in enumerate(branch_iterator(adj, depth=2)):
        text_chain = one_text_chain(chain, abstracts)
        print(f"\n---Chain {i}---")
        for j, text in enumerate(text_chain):
            print(f"   [{j}]: {str(text)[:120]}")
        if i >= 2:
            break

    print("Saving...")
    sp.save_npz(shared_dir / "graph_adj.npz", adj)
    np.save(shared_dir / "pmids.npy", pmids)
    np.save(output_dir / "abstracts.npy", abstracts)
    print("Done.")

main()
