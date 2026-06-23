## library imports
import argparse
import os
import numpy as np 
import json 
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

## local imports


def main():
    cwd = os.getcwd()
    parser = argparse.ArgumentParser()
    # parse input args
    parser.add_argument("--abstractf", required=True, help="File/path/to/abstracts")
    parser.add_argument("--outputd", default=f"{cwd}/embeddings", help="Path/to/output/dir")
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5", help="embedding model directive")
    parser.add_argument("--batch", type=int, default=256, help="size of compute batches")
    parser.add_argument("--chkpt", type=int, default=5, help="checkpoint every x batches")
    args = parser.parse_args()
    # do filesystem stuff
    output_path = os.path.join(args.outputd, "embeddings.npy")
    chkpt_path = f"{cwd}/tmp/checkpoint" 
    os.makedirs(args.outputd, exist_ok=True)
    os.makedirs(f"{cwd}/tmp", exist_ok=True)
    # load abstracts
    abstracts = np.load(args.abstractf, allow_pickle=True)
    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    start_idx = 0
    if os.path.exists(chkpt_path):
        with open(chkpt_path) as f:
                chkpt = json.load(f)
        start_idx = chkpt["last_idx"]
        emb = np.memmap(output_path, dtype='float32', mode='r+', shape=(len(abstracts),dim))
    else:
        emb = np.memmap(output_path, dtype='float32', mode='w+', shape=(len(abstracts),dim))
    for i in tqdm(range(start_idx,len(abstracts),args.batch)):
        batch = abstracts[i:i+args.batch]
        vecs = model.encode(batch, convert_to_numpy = True)
        emb[i:i+len(batch)] = vecs
        if i > start_idx and (i // args.batch) % args.chkpt == 0:
            emb.flush()
            with open(chkpt_path, 'w') as f:
                json.dump({"last_idx": i}, f)
    emb.flush()

if __name__ =="__main__":
    main()
