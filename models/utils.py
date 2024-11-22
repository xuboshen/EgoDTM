from __future__ import division

import logging
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

def inflate_positional_embeds(
    current_model_state_dict,
    new_state_dict,
    num_frames=4,
    load_temporal_fix="bilinear",
):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(current_model_state_dict.keys())
    # import pdb; pdb.set_trace()
    if (
        "visual.temporal_embed" in new_state_dict
        and "visual.temporal_embed" in curr_keys
    ):
        load_temporal_embed = new_state_dict["visual.temporal_embed"]
        load_num_frames = load_temporal_embed.shape[1]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[2]
        # import pdb; pdb.set_trace()
        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(
                    f"### loaded SpaceTimeTransformer model has MORE frames than current..."
                    f"### loading weights, filling in the extras via {load_temporal_fix}"
                )
                new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
            else:
                print(
                    f"### loaded SpaceTimeTransformer model has FEWER frames than current..."
                    f"### loading weights, filling in the extras via {load_temporal_fix}"
                )
                if load_temporal_fix == "zeros":
                    new_temporal_embed = torch.zeros(
                        [load_temporal_embed.shape[0], curr_num_frames, embed_dim]
                    )
                    new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ["interp", "bilinear"]:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = "nearest"
                    if load_temporal_fix == "bilinear":
                        mode = "bilinear"
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(
                        load_temporal_embed, (curr_num_frames, embed_dim), mode=mode
                    ).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict["visual.temporal_embed"] = new_temporal_embed
    elif (
        "visual.base_model.model.temporal_embed" in new_state_dict
        and "visual.base_model.model.temporal_embed" in curr_keys
    ):
        load_temporal_embed = new_state_dict["visual.base_model.model.temporal_embed"]
        load_num_frames = load_temporal_embed.shape[1]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[2]
        # import pdb; pdb.set_trace()
        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(
                    f"### loaded SpaceTimeTransformer model has MORE frames than current..."
                    f"### loading weights, filling in the extras via {load_temporal_fix}"
                )
                new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
            else:
                print(
                    f"### loaded SpaceTimeTransformer model has FEWER frames than current..."
                    f"### loading weights, filling in the extras via {load_temporal_fix}"
                )
                if load_temporal_fix == "zeros":
                    new_temporal_embed = torch.zeros(
                        [load_temporal_embed.shape[0], curr_num_frames, embed_dim]
                    )
                    new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ["interp", "bilinear"]:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = "nearest"
                    if load_temporal_fix == "bilinear":
                        mode = "bilinear"
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(
                        load_temporal_embed, (curr_num_frames, embed_dim), mode=mode
                    ).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict[
                "visual.base_model.model.temporal_embed"
            ] = new_temporal_embed
    # allow loading with smaller spatial patches. assumes custom border crop, to append the
    # border patches to the input sequence
    if "visual.pos_embed" in new_state_dict and "visual.pos_embed" in curr_keys:
        load_pos_embed = new_state_dict["visual.pos_embed"]
        load_num_patches = load_pos_embed.shape[1]
        curr_pos_embed = current_model_state_dict["visual.pos_embed"]
        if load_num_patches != curr_pos_embed.shape[1]:
            raise NotImplementedError(
                "Loading models with different spatial resolution / patch number not yet implemented, sorry."
            )
    elif (
        "visual.base_model.model.pos_embed" in new_state_dict
        and "visual.base_model.model.pos_embed" in curr_keys
    ):
        load_pos_embed = new_state_dict["visual.base_model.model.pos_embed"]
        load_num_patches = load_pos_embed.shape[1]
        curr_pos_embed = current_model_state_dict["visual.base_model.model.pos_embed"]
        if load_num_patches != curr_pos_embed.shape[1]:
            raise NotImplementedError(
                "Loading models with different spatial resolution / patch number not yet implemented, sorry."
            )
    return new_state_dict

def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith("module.") and load_keys[0].startswith(
        "module."
    ):  # this
        undo_dp = True
    elif curr_keys[0].startswith("module.") and not load_keys[0].startswith("module."):
        redo_dp = True

    if undo_dp:  # this
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = "module." + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def load_pretrained_model(cfg, model, model_name, ckpt_path):
    # """load pretrained models from give ckpt_path"""
    # if model_name == "hoi":
    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     state_dict = OrderedDict()
    #     for k, v in ckpt["state_dict"].items():
    #         state_dict[k.replace("module.", "")] = v
    #     model.backbone.logit_scale.requires_grad = False
    #     # inflate weight
    #     print("=> inflating PE in models due to different frame numbers")
    #     state_dict = inflate_positional_embeds(
    #         model.backbone.state_dict(),
    #         state_dict,
    #         num_frames=cfg.INPUT.NUM_FRAMES,
    #         load_temporal_fix="bilinear",
    #     )
    #     keys_to_remove = ["visual.image_projection"]
    #     ktr_state_dict = {
    #         k: v
    #         for k, v in state_dict.items()
    #         if not any(key in k for key in keys_to_remove)
    #     }
    #     new_state_dict = state_dict_data_parallel_fix(ktr_state_dict, state_dict)
    #     model.backbone.load_state_dict(new_state_dict, strict=True)
    #     print(f"=> succesfully loaded pretrained weights from {ckpt_path}")
    # else:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v
    if model_name != 'hoi':
        model.logit_scale.requires_grad = False
    # inflate weight
    print("=> inflating PE in models due to different frame numbers")
    state_dict = inflate_positional_embeds(
        model.state_dict(),
        state_dict,
        num_frames=cfg.INPUT.NUM_FRAMES,
        load_temporal_fix="bilinear",
    )
    model.load_state_dict(state_dict, strict=True)
    print(f"=> succesfully loaded pretrained weights from {ckpt_path}")

    return model
