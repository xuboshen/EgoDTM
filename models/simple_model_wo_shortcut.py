"""
Using FPN only (without DeformableAttn)
Using decoder global embedding for video-text matching, no shortcut for AVION encoder's global embedding
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
# from detectron2.layers import ShapeSpec
from einops import rearrange, repeat
from timm.models.layers import trunc_normal_

from simple_hoi.models.avion import build_avion
from simple_hoi.models.decoder.build import build_decoder
from simple_hoi.models.decoder.fpn import SimpleFeaturePyramid
from .decoder.modules import Conv2d
from .decoder.utils import c2_xavier_fill
from .lavila import build_lavila

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)



class InstanceHOI(nn.Module):
    """InstanceHOI connects video backbone with transformer decoder, together with text encoder
    important args:
        cfg.MODEL.DEC_TYPE: "simple_wo_glob" (local embeddings as kv), 'simple_glob' (use global embeddings as kv)
    """

    def __init__(self, cfg):
        super().__init__()
        self.dec_type = cfg.MODEL.MASK2FORMER.DEC_TYPE
        if cfg.MODEL.BACKBONE == "lavila":
            backbone = build_lavila(cfg)
            decoder = build_decoder(cfg, backbone.vision_width)
        elif cfg.MODEL.BACKBONE == "avion":
            backbone = build_avion(cfg)
            decoder = build_decoder(cfg, in_channel=256, dtype=self.dec_type)
        self.input_proj = nn.Linear(768, 256)
        self.fpn = SimpleFeaturePyramid(
            out_channels=cfg.MODEL.MASK2FORMER.MASK_DIM, scale_factors=(4.0, 1.0)
        )
        conv_dim = cfg.MODEL.MASK2FORMER.CONVS_DIM  # 256
        mask_dim = cfg.MODEL.MASK2FORMER.MASK_DIM  # 768
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        c2_xavier_fill(self.mask_features)
        self.use_videoEcls = cfg.MODEL.USE_VIDEOECLS
        # self.pixel_decoder = build_pixel_decoder(cfg, input_shape={
        #     "p4": ShapeSpec(256, 14, 14, 16),
        #     # "p3": ShapeSpec(256, 28, 28, 8),
        #     "p2": ShapeSpec(256, 56, 56, 4),
        #     # "p1": ShapeSpec(256, 112, 112, 2)
        # })  # input_shape
        # 256: decoder embed dim
        self.use_cap = cfg.MODEL.USE_CAP
        # if self.use_cap:
        #     self.img_query = nn.Parameter(torch.randn(1, 768))
        #     self.img_attn_pool_norm = LayerNorm(768)
        #     self.cross_attention_pooling = CrossAttention(
        #         dim=768, dim_head=768 // 12, heads=12, norm_context=True
        #     )
        if self.use_videoEcls:
            self.image_projection = nn.Parameter(torch.empty(768, 256))
            trunc_normal_(self.image_projection, std=256**-0.5)
        else:
            self.image_projection = nn.Parameter(torch.empty(256, 256))
            trunc_normal_(self.image_projection, std=256**-0.5)
        self.tasks = cfg.TASK.TASK_NAME

        self.backbone = backbone
        # backbone parameters, i.e., lavila forward args
        self.use_checkpoint = cfg.SOLVER.GRADIENT_CHECKPOINT
        self.norm_embed = cfg.MODEL.LAVILA.NORM_EMBED
        self.return_feat = cfg.MODEL.LAVILA.RETURN_FEAT

        # archs
        self.decoder = decoder
        # self.pixel_decoder = pixel_decoder
        # losses
        # self.criterion = criterion
        self.num_queries = cfg.MODEL.MASK2FORMER.NUM_OBJECT_QUERIES
        self.overlap_threshold = cfg.MODEL.MASK2FORMER.TEST.OVERLAP_THRESHOLD
        self.object_mask_threshold = cfg.MODEL.MASK2FORMER.TEST.OBJECT_MASK_THRESHOLD
        # self.metadata = metadata
        if cfg.MODEL.MASK2FORMER.SIZE_DIVISIBILITY < 0:
            # use backbone size_divisibility if not set
            self.size_divisibility = self.backbone.size_divisibility
        else:
            self.size_divisibility = cfg.MODEL.MASK2FORMER.SIZE_DIVISIBILITY
        self.sem_seg_postprocess_before_inference = True
        self.num_frames = cfg.INPUT.NUM_FRAMES
        
        self.vtm_head = nn.Linear(2*256, 2)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, frames, text, **kwargs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        # images = []
        # for video in batched_inputs:
        #     for frame in video["image"]:
        #         images.append(frame.to(self.device))
        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)
        backbone_outputs = self.backbone(frames, text, **kwargs)
        enc_video_global_embed, enc_video_feature_map = (
            backbone_outputs["image_embed"],
            backbone_outputs["image_feat"],
        )
        grid_side = int(((enc_video_feature_map.shape[1]) / 4) ** 0.5)
        # BATCH_SIZE, 4, 196, 768
        video_grid = rearrange(
            enc_video_feature_map,
            "b (t h w) c -> b t (h w) c",
            t=frames.shape[2],
            h=grid_side,
            w=grid_side,
        )
        video_grid = self.input_proj(video_grid)
        # fpn for higher resolutions
        multiscale_features = self.fpn(video_grid)
        # multiscale_features["p4"].shape: torch.Size([8, 768, 14, 14])
        # pixel decoder for multi-scale feature encoding
        mask_features = self.mask_features(multiscale_features["p2"])
        multi_scale_features = [multiscale_features["p4"]]
        # (
        #     mask_features,
        #     transformer_encoder_features,
        #     multi_scale_features,
        # ) = self.pixel_decoder.forward_features(multiscale_features)
        # stop here at first
        dec_outputs = self.decoder(
            multi_scale_features,
            mask_features,  # 3 for multi-scale features, 1 for mask_features, largest
        )  # returns a dict, containing pred_logits and pred_masks
        # import pdb; pdb.set_trace()
        if self.use_cap:
            img_queries = repeat(self.img_query, 'n d -> b n d', b=video_grid.shape[0]).squeeze(1)
            global_image_embedding = self.cross_attention_pooling(
                x=img_queries,
                context=enc_video_feature_map.permute(1, 0, 2),
            ).squeeze(1)
            global_image_embedding = self.img_attn_pool_norm(global_image_embedding)
        else:
            global_image_embedding = dec_outputs["global_embed"].squeeze(0)
        if self.use_videoEcls and not self.use_cap:
            global_image_embedding = enc_video_global_embed
        outputs = {}
        if "VTC" in self.tasks:
            if self.image_projection is not None:
                global_image_embedding = global_image_embedding @ self.image_projection
            outputs.update(
                {
                    "VTC": {
                        "image_embed": F.normalize(global_image_embedding, dim=-1),
                        "video_feature": backbone_outputs["image_feat"],
                        "video_cls": backbone_outputs["image_embed"],
                        "text_embed": backbone_outputs["text_embed"],
                        "logit_scale": backbone_outputs["logit_scale"],
                    }
                }
            )
        if "DepthEstim" in self.tasks:
            outputs.update(
                {
                    "DepthEstim": {
                        "pred_bins": dec_outputs["pred_bins"],
                        "pred_depths": dec_outputs["pred_depths"],
                    },
                    "aux_outputs": dec_outputs["aux_outputs"],
                }
            )

        return outputs
        # if self.training:
        #     # mask classification target
        #     # targets = self.prepare_targets(batched_inputs, images)

        #     # bipartite matching-based loss
        #     losses = self.criterion(outputs, targets)

        #     for k in list(losses.keys()):
        #         if k in self.criterion.weight_dict:
        #             losses[k] *= self.criterion.weight_dict[k]
        #         else:
        #             # remove this loss if not specified in `weight_dict`
        #             losses.pop(k)
        #     return losses
        # else:
        #     mask_cls_results = outputs["pred_logits"]
        #     mask_pred_results = outputs["pred_masks"]

        #     mask_cls_result = mask_cls_results[0]
        #     # upsample masks
        #     mask_pred_result = retry_if_cuda_oom(F.interpolate)(
        #         mask_pred_results[0],
        #         size=(images.tensor.shape[-2], images.tensor.shape[-1]),
        #         mode="bilinear",
        #         align_corners=False,
        #     )

        #     del outputs

        #     input_per_image = batched_inputs[0]
        #     image_size = images.image_sizes[
        #         0
        #     ]  # image size without padding after data augmentation

        #     height = input_per_image.get(
        #         "height", image_size[0]
        #     )  # raw image size before data augmentation
        #     width = input_per_image.get("width", image_size[1])

        #     return retry_if_cuda_oom(self.inference_video)(
        #         mask_cls_result, mask_pred_result, image_size, height, width
        #     )

    def inference_video(
        self, pred_cls, pred_masks, img_size, output_height, output_width
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = (
                torch.arange(self.decoder.num_classes, device=self.device)
                .unsqueeze(0)
                .repeat(self.num_queries, 1)
                .flatten(0, 1)
            )
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.decoder.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks,
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )

            masks = pred_masks > 0.0

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output
