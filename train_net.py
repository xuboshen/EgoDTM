# encoding: utf-8
import argparse
import os
import sys
from collections import OrderedDict

import pandas as pd
import torch


# sys.path.append(".")

from simple_hoi.models import build_model
from simple_hoi.models.utils import load_pretrained_model
from simple_hoi.utils.config_utils import get_cfg


def main():
    # get configs
    args, cfg = get_cfg()
    # components initialization
    model = build_model(cfg)

    # load pretrained models
    if cfg.CKPT_DIR is not None:
        model = load_pretrained_model(cfg, model, 'hoi', cfg.CKPT_DIR)
    import pdb; pdb.set_trace()
    
def load_hoi():
    # get configs
    args, cfg = get_cfg()
    # components initialization
    model = build_model(cfg)

    # load pretrained models
    if cfg.CKPT_DIR is not None:
        model = load_pretrained_model(cfg, model, 'hoi', cfg.CKPT_DIR)
    # import pdb; pdb.set_trace()
    
    return model, 256 # decoder

def load_hoi_backbone():
    # get configs
    args, cfg = get_cfg()
    # components initialization
    model = build_model(cfg)

    # load pretrained models
    if cfg.CKPT_DIR is not None:
        model = load_pretrained_model(cfg, model, 'hoi', cfg.CKPT_DIR)
    # import pdb; pdb.set_trace()
    
    return model, 768 # decoder

if __name__ == "__main__":
    main()
