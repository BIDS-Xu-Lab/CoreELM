import argparse
import json
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import evaluate as hf_evaluate
from openelm.config import load_config
from openelm.model import LlamaForEmbeddingLM, Gemma3ForEmbeddingLM
from openelm.tokens_map import TYPE_TOKEN_MAP_DICT

def load_model(tcfg, output_dir=None):
    checkpoint_dirs = sorted(
        Path(output_dir or tcfg.output_dir).glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1])
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {tcfg.output_dir}")
    checkpoint = str(checkpoint_dirs[-1])
    print(f"Loading checkpoint: {checkpoint}")

    model_config = AutoConfig.from_pretrained(tcfg.basemodel_path)
    if model_config.model_type == "llama":
        model_class = LlamaForEmbeddingLM
    elif model_config.model_type in ["gemma3", "gemma3_text"]:
        model_class = Gemma3ForEmbeddingLM
    else:
        raise ValueError(f"Unsupported model type: {model_config.model_type}")

    elm = model_class.from_pretrained(
        tcfg.basemodel_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch.cuda.current_device()}
    )
    lora_elm = PeftModel.from_pretrained(elm, checkpoint).merge_and_unload()
    lora_elm.eval()

    tokenizer = AutoTokenizer.from_pretrained(tcfg.basemodel_path)
    gen_tok_id = TYPE_TOKEN_MAP_DICT[model_config.model_type]["gen_tok_id"]
    return lora_elm, tokenizer, gen_tok_id

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained ctELM graph model.")
    parser.add_argument("--config", default="configs/pipeline.yaml")
    parser.add_argument("--variant", default=None)
    parser.add_argument("--experiment", default=None)
    args = parser.parse_args()

    cfg  = load_config(args.config, args.variant, args.experiment)
    ecfg = cfg.eval
    tcfg = cfg.train

    prefix = getattr(cfg.paths, 'experiment_prefix', '')
    if prefix:
        p = Path(tcfg.output_dir)
        output_dir = str(p.parent / prefix / p.name)
    else:
        output_dir = tcfg.output_dir

    graph_outputd      = Path(cfg.paths.graph_outputd)
    embeddings_outputd = Path(cfg.paths.embeddings_outputd)
    dataset_outputd    = graph_outputd / cfg.paths.dataset_subdir

    lora_elm, tokenizer, gen_tok_id = load_model(tcfg, output_dir=output_dir)

    abstracts  = np.load(graph_outputd / "abstracts.npy", allow_pickle=True)
    embeddings = np.memmap(
        embeddings_outputd / "embeddings.npy",
        dtype="float32", mode="r", shape=(len(abstracts), cfg.embed_abstracts.embed_dim)
    )

    val_ds       = Dataset.load_from_disk(str(dataset_outputd / "evaluation"))
    embed_model  = SentenceTransformer(cfg.embed_abstracts.model)
    bertscore    = hf_evaluate.load("bertscore")

    results    = []
    batch_size = ecfg.batch_size

    for batch_start in range(0, len(val_ds), batch_size):
        batch = val_ds[batch_start:batch_start + batch_size]

        # Build prompt-only input_ids (everything before gen_tok)
        prompt_tensors = []
        for ids in batch["input_ids"]:
            gen_pos = ids.index(gen_tok_id)
            prompt_tensors.append(torch.tensor(ids[:gen_pos], dtype=torch.long))

        max_len = max(t.size(0) for t in prompt_tensors)
        padded  = torch.full((len(prompt_tensors), max_len), tokenizer.pad_token_id, dtype=torch.long)
        prompt_lengths = []
        for i, t in enumerate(prompt_tensors):
            padded[i, :t.size(0)] = t
            prompt_lengths.append(t.size(0))
        padded = padded.to("cuda")

        # Flatten domain_embeddings: [ex0_emb0, ex0_emb1, ex1_emb0, ...]
        domain_embs = [
            torch.tensor(e, dtype=torch.bfloat16).to("cuda")
            for embs in batch["domain_embeddings"]
            for e in embs
        ]

        with torch.no_grad():
            outputs = lora_elm.generate(
                input_ids=padded,
                domain_embeddings=domain_embs,
                max_new_tokens=ecfg.max_new_tokens,
                eos_token_id=lora_elm.config.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                repetition_penalty=ecfg.repetition_penalty,
            )

        for j, (output, prompt_len) in enumerate(zip(outputs, prompt_lengths)):
            generated   = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)
            target_idx  = batch["target_idx"][j]
            target_text = str(abstracts[target_idx])

            gen_emb    = embed_model.encode(generated, convert_to_numpy=True)
            target_emb = np.array(embeddings[target_idx])
            cos_sim    = float(np.dot(gen_emb, target_emb) / (np.linalg.norm(gen_emb) * np.linalg.norm(target_emb) + 1e-8))

            results.append({
                "target_idx":        int(target_idx),
                "generated":         generated,
                "target_text":       target_text,
                "cosine_similarity": cos_sim,
            })

        print(f"  {min(batch_start + batch_size, len(val_ds))}/{len(val_ds)} examples evaluated")

    # BERTScore computed in one pass over all results
    print("Computing BERTScore...")
    bs = bertscore.compute(
        predictions=[r["generated"]    for r in results],
        references= [r["target_text"]  for r in results],
        lang="en"
    )
    for i, r in enumerate(results):
        r["bertscore_precision"] = bs["precision"][i]
        r["bertscore_recall"]    = bs["recall"][i]
        r["bertscore_f1"]        = bs["f1"][i]

    cos_sims = [r["cosine_similarity"] for r in results]
    bs_f1s   = [r["bertscore_f1"]      for r in results]
    summary  = {
        "n": len(results),
        "cosine_similarity": {"mean": float(np.mean(cos_sims)), "std": float(np.std(cos_sims))},
        "bertscore_f1":      {"mean": float(np.mean(bs_f1s)),   "std": float(np.std(bs_f1s))},
    }

    output_path = Path(output_dir) / "eval_results.json"
    with open(output_path, "w") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2)

    print(f"\n=== Evaluation Results ===")
    print(f"N:                  {summary['n']}")
    print(f"Cosine Similarity:  {summary['cosine_similarity']['mean']:.4f} ± {summary['cosine_similarity']['std']:.4f}")
    print(f"BERTScore F1:       {summary['bertscore_f1']['mean']:.4f} ± {summary['bertscore_f1']['std']:.4f}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
