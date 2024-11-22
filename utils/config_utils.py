import argparse
import os

from simple_hoi.base import cfg


def get_cfg():
    parser = argparse.ArgumentParser(description="PyTorch Template MNIST Training")
    parser.add_argument(
        "--config-file", default="", help="path to config file", type=str
    )

    parser.add_argument(
        "--train-anno-file",
        default="",
        help="path to training annotation file",
        type=str,
    )
    parser.add_argument("--use-rich-text", action="store_true")

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()

    return args, cfg
