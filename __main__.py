## Library imports  
import argparse
import numpy as np 
import scipy.sparse as sp 
from pathlib import Path

## Internal imports
from .build import load_pmids, load_abstracts, fetch_citations, build_edges, build_csr

parser = argparse.ArgumentParser(description = 'Build Citation DAG from icite data.')   
parser.add_argument('--txt', required=True, help='Abstracts raw text file')
parser.add_argument('--pmidf', required=True, help='Ordered PMIDs, one per line')
parser.add_argument('--db', default = 'data/icite/icite.db', help = 'Path to citation database')
parser.add_argument('--outputd', default='./graph_output', help='Directory to write output files (created if absent)')
args = parser.parse_args()

output_dir = Path(args.outputd)
output_dir.mkdir(parents=True, exist_ok=True)

print('Loading PMIDs...')
pmids, pmid_idx = load_pmids(args.pmidf)
print(f'  {len(pmids)} PMIDs loaded')

print('Loading abstracts...')
abstracts = load_abstracts(args.txt, pmid_idx)
print(f'  {len(abstracts)} PMIDs loaded')

print('Fetching citations...')
pmid_tbl = fetch_citations(args.db, pmid_idx)
print(f'  {len(pmid_tbl)} rows returned')

print('Building edges...')
edges = build_edges(pmid_tbl, pmid_idx)
print(f'  {len(edges)} edges found')

print('Building CSR...')
adj = build_csr(edges, len(pmids))

print('Saving...')
sp.save_npz(output_dir / 'graph_adj.npz', adj)
np.save(output_dir / 'pmids.npy', pmids)
np.save(output_dir / 'abstracts.npy', abstracts)
print('Done.')
