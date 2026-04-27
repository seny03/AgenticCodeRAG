import datasets.config
import datasets.utils.logging as _ds_logging
from datasets import load_dataset as _hf_load_dataset


def load_dataset_cached(name: str, split: str, **kwargs):
    """Load an HF dataset from local cache; fall back to network download on cache miss."""
    prev_offline = datasets.config.HF_DATASETS_OFFLINE
    prev_verbosity = _ds_logging.get_verbosity()
    datasets.config.HF_DATASETS_OFFLINE = True
    _ds_logging.set_verbosity_error()
    try:
        return _hf_load_dataset(name, split=split, **kwargs)
    except Exception:
        datasets.config.HF_DATASETS_OFFLINE = False
        _ds_logging.set_verbosity(prev_verbosity)
        return _hf_load_dataset(name, split=split, **kwargs)
    finally:
        datasets.config.HF_DATASETS_OFFLINE = prev_offline
        _ds_logging.set_verbosity(prev_verbosity)
