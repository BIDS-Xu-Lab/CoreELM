from omegaconf import OmegaConf

def load_config(config_path="configs/pipeline.yaml", experiment_path=None):
    cfg = OmegaConf.load(config_path)
    if experiment_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(experiment_path))
    return cfg
