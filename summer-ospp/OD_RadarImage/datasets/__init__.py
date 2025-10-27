from typing import Any, Dict

# from datasets.loader import load_listed
from datasets.kradar.dataset import initialize_kradar
from datasets.kradar.processor import prepare_kradar

def prepare(dataset: str, *args: Any, **kwargs: Any):
    if dataset == "kradar":
        return prepare_kradar(*args, **kwargs)
    raise ValueError(f"Dataset {dataset} is not supported!")


def init(dataset: str, *args: Any, **kwargs: Any):
    if dataset == "kradar":
        return initialize_kradar(*args, **kwargs)
    raise ValueError(f"Dataset {dataset} is not supported!")


def load(dataset, config: Dict[str, Any], *args, **kwargs):
    return load_listed(dataset, config, *args, **kwargs)