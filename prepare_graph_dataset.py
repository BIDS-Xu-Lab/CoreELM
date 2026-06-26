import argparse
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import Dataset, interleave_datasets
from openelm.tokens_map import TOKEN_MAP_DICT
from openelm.graph.traverse import branch_iterator
from openelm.config import load_config

def graph_chain_generator(adj, abstracts, embeddings, tokenizer, emb_token, gen_token, depth, max_chains=None, task=None):
    n_slots = task.prompt_template.count("{emb_token}") if task else 0
    for i, chain in enumerate(branch_iterator(adj, depth=depth)):
        if max_chains is not None and i >= max_chains:
            break
        if n_slots > 1 and len(chain) != n_slots:
            continue
        target = abstracts[chain[-1]]
        if target is None:
            continue
        if n_slots == 1:
            prompt = task.prompt_template.replace("{emb_token}", " ".join([emb_token] * len(chain)))
        elif n_slots > 1:
            prompt = task.prompt_template.format(emb_token=emb_token)
        else:
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

    tasks = pcfg.get("tasks", None)
    task_list = list(tasks) if tasks else [None]

    print("Building dataset from citation chains...")
    train_datasets = []
    val_datasets   = []
    for task in task_list:
        ds = Dataset.from_generator(
            lambda t=task: graph_chain_generator(
                adj, abstracts, embeddings, tokenizer,
                emb_token, gen_token, pcfg.depth, pcfg.max_chains, t
            )
        )
        split = ds.train_test_split(test_size=1 - pcfg.train_ratio, seed=42)
        train_datasets.append(split["train"])
        val_datasets.append(split["test"])

    train_ds = interleave_datasets(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds   = interleave_datasets(val_datasets)   if len(val_datasets)   > 1 else val_datasets[0]

    train_ds.save_to_disk(str(dataset_outputd / "train"))
    val_ds.save_to_disk(str(dataset_outputd / "validation"))

    print(f"Done. {len(train_ds)} train / {len(val_ds)} validation chains saved to {dataset_outputd}")

if __name__ == "__main__":
    main()
