from .botanic_garden import BotanicGarden
from .digiforests import DigiForests
from .eth3d import ETH3d
from .poly_tunnel import PolyTunnel
from .tartan_air import TartanAir


DATASET_REGISTRY = {
    "BotanicGarden": BotanicGarden,
    "TartanAir": TartanAir,
    "PolyTunnel": PolyTunnel,
    "DigiForests": DigiForests,
    "ETH3d": ETH3d,
}


def infer_dataset_name(config_path):
    for dataset_name in DATASET_REGISTRY:
        if dataset_name in str(config_path):
            return dataset_name
    raise ValueError(f"Cannot infer dataset type from config path: {config_path}")


def build_dataset(configs, config_path=None, dataset_name=None):
    selected_name = dataset_name or infer_dataset_name(config_path)
    return DATASET_REGISTRY[selected_name](configs)
