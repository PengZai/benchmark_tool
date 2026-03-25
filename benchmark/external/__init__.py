import importlib.util
from omegaconf import DictConfig, OmegaConf
import numpy as np


def resolve_special_float(value):
    if value == "inf":
        return np.inf
    elif value == "-inf":
        return -np.inf
    else:
        raise ValueError(f"Unknown special float value: {value}")


# Define model configurations with import paths
MODEL_CONFIGS = {
    "priorda": {
        "module": "benchmark.external.priorda",
        "class_name": "PriorDepthAnythingWrapper",
    },
    "mapanything": {
        "module": "benchmark.external.mapanything",
        "class_name": "MapAnythingWrapper",
    },
    "mast3r": {
        "module": "benchmark.external.mast3r",
        "class_name": "MASt3RSGWrapper",
    },
    "multi_view_stereo": {
        "module": "benchmark.external.multi_view_stereo",
        "class_name": "MVSWrapper",
    },
    "depth_anything_v2": {
        "module": "benchmark.external.depth_anything_v2",
        "class_name": "DepthAnythingV2Wrapper",
    },
}


def model_factory(model_str: str, **kwargs):

    model_config = MODEL_CONFIGS[model_str]
    module_path = model_config["module"]
    class_name = model_config["class_name"]


    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    return model_class(**kwargs)


def init_model(
    model_str: str, model_config: DictConfig, torch_hub_force_reload: bool = False
):
    """
    Initialize a model using OmegaConf configuration.

    Args:
        model_str (str): Name of the model class to create.
        model_config (DictConfig): OmegaConf model configuration.
        torch_hub_force_reload (bool): Whether to force reload relevant parts of the model from torch hub.
    """
    if not OmegaConf.has_resolver("special_float"):
        OmegaConf.register_new_resolver("special_float", resolve_special_float)
    model_dict = OmegaConf.to_container(model_config, resolve=True)
    model = model_factory(
        model_str, torch_hub_force_reload=torch_hub_force_reload, **model_dict
    )

    return model


