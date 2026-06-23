# ctELM Architecture

## 1. Model Class Hierarchy (`openelm/model.py`)

```mermaid
classDiagram
    class LlamaConfig {
        <<HuggingFace>>
    }
    class Gemma3TextConfig {
        <<HuggingFace>>
    }
    class LlamaForCausalLM {
        <<HuggingFace>>
    }
    class Gemma3ForCausalLM {
        <<HuggingFace>>
    }

    class EmbeddingLMConfigMixin {
        +dim_embed_domain : int
        +dim_adapter_hidden : int
        +pretrained_model_name_or_path : str
    }

    class LlamaForEmbeddingConfig {
        +model_type = "llama"
    }
    class Gemma3ForEmbeddingConfig {
        +model_type = "gemma3"
    }

    class EmbeddingLMMixin {
        +adapter : Sequential
        +emb_tok_id : int
        +gen_tok_id : int
        +forward(input_ids, domain_embeddings)
        +prepare_inputs_for_generation(...)
    }

    class LlamaForEmbeddingLM
    class Gemma3ForEmbeddingLM

    LlamaConfig <|-- LlamaForEmbeddingConfig
    EmbeddingLMConfigMixin <|-- LlamaForEmbeddingConfig

    EmbeddingLMConfigMixin <|-- Gemma3ForEmbeddingConfig
    Gemma3TextConfig <|-- Gemma3ForEmbeddingConfig

    EmbeddingLMMixin <|-- LlamaForEmbeddingLM
    LlamaForCausalLM <|-- LlamaForEmbeddingLM

    EmbeddingLMMixin <|-- Gemma3ForEmbeddingLM
    Gemma3ForCausalLM <|-- Gemma3ForEmbeddingLM

    LlamaForEmbeddingLM ..> LlamaForEmbeddingConfig : uses
    Gemma3ForEmbeddingLM ..> Gemma3ForEmbeddingConfig : uses
```

The `adapter` inside `EmbeddingLMMixin` is:
`Linear(domain_dim → adapter_hidden) → ReLU → Linear(adapter_hidden → token_dim)`

In `forward()`, any position where `input_ids == emb_tok_id` has its token embedding replaced by `adapter(domain_embedding[i])` before being passed to the transformer layers.

---

## 2. Graph Module Internals (`openelm/graph/`)

```mermaid
graph LR
    subgraph graph["openelm/graph/"]
        build["build.py
load_pmids()
load_abstracts()
fetch_citations()
build_edges()
build_csr()"]

        traverse["traverse.py
leaves()
walk_from_leaf()
branch_iterator()"]

        chains["chains.py
one_text_chain()"]

        main["__main__.py
CLI entry point"]
    end

    regex["openelm/regex_parse.py
extract_abstracts()"]

    main --> build
    main --> traverse
    main --> chains
    traverse -.->|CSR adj| build
    chains --> traverse
    build --> regex
```

---

## 3. End-to-End Data & Training Pipeline

```mermaid
flowchart TD
    subgraph ext["External Inputs"]
        PubMed[(PubMed200K RCT\nabstracts + PMIDs)]
        iCite[(iCite SQLite DB\n37.8M papers)]
        BaseLM[(Base LM\nLlama-3 or Gemma-3)]
    end

    subgraph s1["Step 1 — Build Citation Graph\n`python -m openelm.graph`"]
        GB["build.py\nfetch_citations → build_edges → build_csr\n190,654 nodes · 645,877 edges"]
    end

    PubMed --> GB
    iCite --> GB

    GB --> adj[(graph_output/graph_adj.npz\nCSR 190k × 190k)]
    GB --> abs[(graph_output/abstracts.npy)]
    GB --> pids[(graph_output/pmids.npy)]

    subgraph s2["Step 2 — Encode Abstracts\n`embed_abstracts.py`"]
        EA["SentenceTransformer\nbge-large-en-v1.5\nbatch encode + memmap checkpoint"]
    end

    abs --> EA
    EA --> emb[(embeddings/embeddings.npy\n190,654 × 1024 · float32 memmap)]

    subgraph s3["Step 3 — Initialize ELM\n`initialize_model.py`"]
        IM["initialize_embedding_model_from_causal_lm()\ncopies LM weights · randomly inits adapter"]
    end

    BaseLM --> IM
    IM --> ckpt[(initial_elm_model/\nELM checkpoint + tokenizer)]

    subgraph s4["Step 4 — Prepare Dataset"]
        PD["prepare_dataset.py\nYAML config · static embeddings\ntask_specific_generator()"]
        PGD["prepare_graph_dataset.py\ncitation-chain generator\ngraph_chain_generator()"]
    end

    PubMed -.->|embeddings + texts via YAML| PD
    adj --> PGD
    abs --> PGD
    emb --> PGD

    PD  --> hfo[(HF Dataset\nstatic train / val)]
    PGD --> hfg[(HF Dataset\ngraph train / val\n4,749 / 251 chains)]

    subgraph s5["Step 5 — Train\n`train.py`"]
        TR["SFTTrainer + LoRA\ntarget: q_proj · k_proj\nmodules_to_save: adapter"]
    end

    ckpt --> TR
    hfo --> TR
    hfg --> TR
    TR --> trained[(Trained ELM\nLoRA + adapter checkpoint)]

    subgraph s6["Step 6 — Inference\n`inference.py`"]
        INF["load_elm_model()\nbatched_inference_input_generator()\nlora_elm.generate()"]
    end

    trained --> INF
    emb --> INF
    INF --> out([Generated text\n*.pkl per task])
```
