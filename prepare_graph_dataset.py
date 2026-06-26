import argparse
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset
from openelm.tokens_map import TOKEN_MAP_DICT
from openelm.graph.traverse import branch_iterator
from openelm.config import load_config

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

def main():
    parser = argparse.ArgumentParser(description="Prepare graph chain dataset for ctELM.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.experiment)
    pcfg = cfg.prepare_graph_dataset

    if pcfg.base_model not in TOKEN_MAP_DICT:
        raise ValueError(f"base_model '{pcfg.base_model}' not in TOKEN_MAP_DICT")

    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    dataset_outputd    = graph_outputd / cfg.paths.dataset_subdir
    dataset_outputd.mkdir(parents=True, exist_ok=True)

    print("Loading graph...")
    adj = sp.load_npz(graph_outputd / "graph_adj.npz")

    print("Loading abstracts...")
    abstracts = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)

    print("Loading embeddings...")
    embeddings = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(len(abstracts), cfg.embed_abstracts.embed_dim)
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(pcfg.base_model)
    emb_token = TOKEN_MAP_DICT[pcfg.base_model]["emb_tok"]
    gen_token  = TOKEN_MAP_DICT[pcfg.base_model]["gen_tok"]

    print("Building dataset from citation chains...")
    dataset = Dataset.from_generator(
        lambda: graph_chain_generator(
            adj, abstracts, embeddings, tokenizer,
            emb_token, gen_token, pcfg.depth, pcfg.max_chains
        )
    )

    split = dataset.train_test_split(test_size=1 - pcfg.train_ratio, seed=42)
    split["train"].save_to_disk(str(dataset_outputd / "train"))
    split["test"].save_to_disk(str(dataset_outputd / "validation"))

    print(f"Done. {len(split['train'])} train / {len(split['test'])} validation chains saved to {dataset_outputd}")

if __name__ == "__main__":
    main()
