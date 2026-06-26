print("""
ctELM graph-time embedding pipeline
------------------------------------
  python -m openelm.graph            # build citation DAG
  python embed_abstracts.py          # encode abstracts with sentence transformer
  python prepare_graph_dataset.py    # build HF dataset from citation chains

All scripts accept:
  --config     path to pipeline YAML  (default: configs/pipeline.yaml)
  --experiment path to experiment override YAML (optional)
""")
