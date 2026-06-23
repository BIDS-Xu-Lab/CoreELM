## library imports
import os
import argparse
import numpy as np
import scipy.sparse as sp
import torch
from transformers import AutoTokenizer
from datasets import Dataset
from openelm.tokens_map import TOKEN_MAP_DICT
from openelm.graph.traverse import branch_iterator


def graph_chain_generator(adj, abstracts, embeddings, tokenizer, emb_token, gen_token, depth, max_chains=None):
    for i, chain in enumerate(branch_iterator(adj, depth=depth)):
        if max_chains is not None and i >= max_chains:
            break
        target = abstracts[chain[-1]]
        if target is None:
            continue
        prompt = " ".join([emb_token] * len(chain))
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gen_token + str(target)},
        ]
        domain_embeddings = [torch.Tensor(embeddings[idx]) for idx in chain]
        yield {
            "input_ids": tokenizer.apply_chat_template(chat),
            "domain_embeddings": domain_embeddings,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare graph chain dataset for ctELM.")
    parser.add_argument("--adj",        required=True,  help="Path to graph_adj.npz")
    parser.add_argument("--abstracts",  required=True,  help="Path to abstracts.npy")
    parser.add_argument("--embeddings", required=True,  help="Path to embeddings.npy (raw memmap)")
    parser.add_argument("--outputd",    required=True,  help="Directory to write HF dataset")
    parser.add_argument("--base_model", required=True,  help="Base model name (for tokenizer)")
    parser.add_argument("--depth",      type=int, default=5, help="Citation chain depth")
    parser.add_argument("--train_ratio",type=float, default=0.95, help="Fraction of chains for training")
    parser.add_argument("--max_chains", type=int, default=None, help="Cap number of chains (default: all)")
    args = parser.parse_args()

    if args.base_model not in TOKEN_MAP_DICT:
        raise ValueError(f"base_model '{args.base_model}' not in TOKEN_MAP_DICT")

    os.makedirs(args.outputd, exist_ok=True)

    print("Loading graph...")
    adj = sp.load_npz(args.adj)

    print("Loading abstracts...")
    abstracts = np.load(args.abstracts, allow_pickle=True)

    print("Loading embeddings...")
    embeddings = np.memmap(args.embeddings, dtype="float32", mode="r", shape=(len(abstracts), 1024))

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    emb_token = TOKEN_MAP_DICT[args.base_model]["emb_tok"]
    gen_token  = TOKEN_MAP_DICT[args.base_model]["gen_tok"]

    print("Building dataset from citation chains...")
    dataset = Dataset.from_generator(
        lambda: graph_chain_generator(adj, abstracts, embeddings, tokenizer, emb_token, gen_token, args.depth, args.max_chains)
    )

    split = dataset.train_test_split(test_size=1 - args.train_ratio, seed=42)

    split["train"].save_to_disk(os.path.join(args.outputd, "train"))
    split["test"].save_to_disk(os.path.join(args.outputd, "validation"))

    print(f"Done. {len(split['train'])} train / {len(split['test'])} validation chains saved to {args.outputd}")
