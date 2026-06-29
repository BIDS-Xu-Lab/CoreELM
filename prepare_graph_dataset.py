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

def graph_chain_generator(adj, abstracts, embeddings, tokenizer, emb_token, gen_token, depth, n_total, seed, task=None):
    n_slots = task.prompt_template.count("{emb_token}") if task else 0
    include_target = task.get("include_target_embedding", True) if task else True
    # chain length = n_slots when target is included, n_slots+1 when withheld
    expected_chain_len = n_slots if include_target else n_slots + 1
    for chain in branch_iterator(adj, depth=depth, max_chains=n_total, seed=seed):
        if n_slots > 0 and len(chain) != expected_chain_len:
            continue
        target = abstracts[chain[-1]]
        if target is None:
            continue
        if n_slots > 0:
            prompt = task.prompt_template.format(emb_token=emb_token)
        else:
            indices_count = len(chain) if include_target else len(chain) - 1
            prompt = " ".join([emb_token] * indices_count)
        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": gen_token + str(target)},
        ]
        indices = chain if include_target else chain[:-1]
        domain_embeddings = [torch.Tensor(embeddings[idx]) for idx in indices]
        yield {
            "input_ids": tokenizer.apply_chat_template(chat),
            "domain_embeddings": domain_embeddings,
            "target_idx": int(chain[-1]),
        }

def main():
    parser = argparse.ArgumentParser(description="Prepare graph chain dataset for ctELM.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.variant, args.experiment)
    pcfg = cfg.prepare_graph_dataset

    if pcfg.base_model not in TOKEN_MAP_DICT:
        raise ValueError(f"base_model '{pcfg.base_model}' not in TOKEN_MAP_DICT")

    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    dataset_outputd    = graph_outputd / cfg.paths.dataset_subdir
    dataset_outputd.mkdir(parents=True, exist_ok=True)

    n_total = pcfg.n_train + pcfg.n_val + pcfg.n_eval

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

    tasks     = pcfg.get("tasks", None)
    task_list = list(tasks) if tasks else [None]

    print(f"Sampling {n_total} chains per task ({len(task_list)} task(s))...")
    train_datasets = []
    val_datasets   = []
    eval_datasets  = []

    for task in task_list:
        ds = Dataset.from_generator(
            lambda t=task: graph_chain_generator(
                adj, abstracts, embeddings, tokenizer,
                emb_token, gen_token, pcfg.depth, n_total, pcfg.seed, t
            )
        )
        ds = ds.shuffle(seed=pcfg.seed)
        train_datasets.append(ds.select(range(pcfg.n_train)))
        val_datasets.append(  ds.select(range(pcfg.n_train, pcfg.n_train + pcfg.n_val)))
        eval_datasets.append( ds.select(range(pcfg.n_train + pcfg.n_val, n_total)))

    def merge(dsets):
        return interleave_datasets(dsets) if len(dsets) > 1 else dsets[0]

    train_ds = merge(train_datasets)
    val_ds   = merge(val_datasets)
    eval_ds  = merge(eval_datasets)

    train_ds.save_to_disk(str(dataset_outputd / "train"))
    val_ds.save_to_disk(  str(dataset_outputd / "validation"))
    eval_ds.save_to_disk( str(dataset_outputd / "evaluation"))

    print(f"Done. {len(train_ds)} train / {len(val_ds)} validation / {len(eval_ds)} evaluation chains saved to {dataset_outputd}")

if __name__ == "__main__":
    main()
