import argparse
import numpy as np 
import scipy.sparse as sp 
import sqlite3
from pathlib import Path

'''
    data pipeline script to construct a graph relationship between citations based on PMIDs
'''

def load_pmids(path):
    pmids = np.loadtxt(path, dtype=np.int64)
    pmid_idx = {int(p): i for i, p in enumerate(pmids)}
    return pmids, pmid_idx

def fetch_citations(db_path, pmid_idx):
    '''
        reads in list of citations from pmid
        use a temp table to fetch cited_by from icite.db 
        return [($cited_pmid, "$citing_pmid $citing_pmid ... "), ...]
    '''
    src = pmid_idx.keys()
    pmids_rows = zip(src)
    db = sqlite3.connect(db_path)
    curs = db.cursor()
    pmid_table = curs.execute("CREATE TEMP TABLE temp_pmids(pmid INTEGER PRIMARY KEY"))
    curs.executemany("INSERT INTO temp_pmids VALUES(?)", pmids_rows)
    curs.execute('''SELECT p.pmid, p.cited_by
                    FROM papers p
                    JOIN temp_pmids t ON p.pmid = t.pmid
                    WHERE p.cited_by IS NOT NULL AND p.cited_by != ""''')
    # pmid_tbl = [($cited_pmid, "$citing_pmid $citing_pmid")]
    return curs.fetchall()

def build_edges(pmid_tbl, pmid_idx, weight_fn=None):
    '''
        build a list of all edges in db
        returns [(cited_idx, citing_idx, weight), ...]
        weight_fn(cited_pmid, citing_pmid) -> float, defaults to 1.0
    '''
    edges = []
    for row in pmid_tbl:
        cited_pmid = int(row[0])
        citers = [int(i) for i in row[1].split(" ") if int(i) in pmid_idx]
        for citing_pmid in citers:
            w = weight_fn(cited_pmid, citing_pmid) if weight_fn else 1.0
            edges.append((pmid_idx[cited_pmid], pmid_idx[citing_pmid], w))
    return edges

def build_csr(edges, n):
    rows, cols, data = zip(*edges)
    return sp.csr_matrix(
        (np.array(data, dtype=np.float32), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
        shape=(n, n)
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser(description = 'Build Citation DAG from icite data.')
    parser.add_argument('--pmidf', required=True, help='Ordered PMIDs, one per line')
    parser.add_argument('--db', default = '../data/icite/icite.db', help = 'Path to citation database')
    parser.add_argument('--outputd', required=True, help='Directory to write output files')
    args = parser.parse_args()

    output_dir = Path(args.outputd)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Loading PMIDs...')
    pmids, pmid_idx = load_pmids(args.pmidf)
    print(f'  {len(pmids)} PMIDs loaded')

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
    print('Done.')
