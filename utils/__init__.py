import keyword
import logging
import dataclasses
from pathlib import Path


class Config(object):
    """A config class that supports attribute access."""

    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if k in keyword.kwlist or not k.isidentifier():
                raise ValueError("config key `{}` is invalid!".format(k))
            if hasattr(self, k):
                raise ValueError("config key `{}` already exists!".format(k))
            if v in keyword.kwlist:
                raise ValueError("config value `{}` is a python keyword!".format(v))
            setattr(self, k, v)


def increment_path(path, sep="-"):
    """ Automatically increment path, i.e. weights/exp -> weights/exp{sep}2, weights/exp{sep}3, ..."""
    path = Path(path)
    if path.exists():
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not Path(p).exists():
                path = Path(p)
                break
        path.mkdir(parents=True, exist_ok=True)  # make directory
    else:
        path.mkdir(parents=True, exist_ok=True)
    return path


def get_logger(name, save_dir, enable_ch=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    ch.flush()
    fh = logging.FileHandler(Path(save_dir, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    if enable_ch:
        logger.addHandler(ch)
    logger.addHandler(fh)
    return logger
