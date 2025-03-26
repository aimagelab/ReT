import torch
from transformers import logging
import os

logging.set_verbosity_info()
logging.enable_explicit_format()


def get_logger():
    return logging.get_logger("transformers")


def get_additive_attn_mask(binary_attn_mask, dtype):
    ret = torch.where(binary_attn_mask.bool(), 0, torch.finfo(dtype).min).to(dtype)
    return ret


def is_debug():
    return int(os.getenv("DEBUG", 0)) == 1