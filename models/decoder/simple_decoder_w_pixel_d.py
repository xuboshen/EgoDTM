import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn import functional as F

from .modules import MLP, Conv2d, CrossAttentionLayer, FFNLayer, SelfAttentionLayer
from .position_encoding import PositionEmbeddingSine3D
from .utils import c2_xavier_fill, check_if_dynamo_compiling


class MaskedTransformerDecoder(nn.Module):
    def __init__(
        self,
        cfg,
        in_channels,
        mask_classification=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """

        super().__init__()
        num_classes = cfg.MODEL.MASK2FORMER.NUM_CLASSES
        hidden_dim = cfg.MODEL.MASK2FORMER.HIDDEN_DIM
        num_queries = cfg.MODEL.MASK2FORMER.NUM_OBJECT_QUERIES
        nheads = cfg.MODEL.MASK2FORMER.NUM_HEADS
        dim_feedforward = cfg.MODEL.MASK2FORMER.DIM_FEEDFORWARD
        dec_layers = cfg.MODEL.MASK2FORMER.DEC_LAYERS - 1
        pre_norm = cfg.MODEL.MASK2FORMER.PRE_NORM
        enforce_input_project = cfg.MODEL.MASK2FORMER.ENFORCE_INPUT_PROJ
        mask_dim = cfg.MODEL.MASK2FORMER.MASK_DIM
        num_frames = cfg.INPUT.NUM_FRAMES
        self.use_flash_attn = cfg.MODEL.MASK2FORMER.USE_FLASH_ATTN
        # assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        self.num_frames = num_frames

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)

        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    use_flash_attn=self.use_flash_attn,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                    use_flash_attn=self.use_flash_attn,
                )
            )

        self.num_classes = num_classes
        # self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries + 1, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries + 1, hidden_dim)

        # level embedding (we always use 3 scales, use 1 for trial)
        self.num_feature_levels = 3
        # self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            pass
            # self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.class_embed = nn.Linear(hidden_dim, 1)
        # self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        # self.image_projection = nn.Parameter(
        #     torch.empty(hidden_dim, cfg.MODEL.LAVILA.PROJECT_EMBED_DIM)
        # )
        # nn.init.normal_(self.image_projection, std=hidden_dim**-0.5)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    def forward(self, x, mask_features, mask=None):
        """
        args:
            x: multi-scale feature (bT, C, H, W), used as (k, v) pairs
            mask_features: initialized by video final feature, used to generate mask for (k, v)
                (bT, C, H, W) # 1, 4, 196, 768
        returns:
            "pred_logits": predictions_class[-1] from the last layer, shape (BATCH_SIZE, 100, 3)
            "pred_masks": predictions_depth[-1], shape [2, 100, 4, 14, 14]
            "aux_outputs": self._set_aux_loss(
                predictions_class if self.mask_classification else None,
                predictions_depth,
            ), List[dicts(pred_logits=(2, 100, 3), pred_masks=(2, 100, 4, 14, 14))], e.g., 9 items
        """
        mask_features = rearrange(mask_features, "(b t) c h w -> b t c h w", t=4)
        bs, t = mask_features.shape[:2]

        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        # preparing (k, v) pairs
        for i in range(self.num_feature_levels):
            # x[i] = rearrange(
            #     x[i], "(b t) c h w -> (b t) c h w", h=mask_features.shape[-1]
            # )
            size_list.append(x[i].shape[-2:])
            src.append(self.input_proj[i](x[i]).flatten(2))
            # BSxCxHW => BSxTxCxHW => (TxHW)xBSxC
            _, c, hw = src[-1].shape
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        src_flatten = torch.cat(src, dim=0)  # sum(THW), BS, C
        # QxNxC positional embedding
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # query themselves
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        # predictions_class = []
        # predictions_depth = []

        # # prediction heads on learnable query features
        # outputs_class, outputs_depth, attn_mask = self.forward_prediction_heads(
        #     output, mask_features, attn_mask_target_size=size_list[0]
        # )  # attn_mask: [24, 17, 784]
        # predictions_class.append(outputs_class)
        # predictions_depth.append(outputs_depth)
        for i in range(self.num_layers):
            # level_index = i % self.num_feature_levels
            # attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output,
                src_flatten,
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                query_pos=query_embed,
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None, tgt_key_padding_mask=None, query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](output)

            # outputs_class, outputs_depth, attn_mask = self.forward_prediction_heads(
            #     output,
            #     mask_features,
            #     attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels],
            # )
            # predictions_class.append(outputs_class)
            # predictions_depth.append(outputs_depth)
        # assert len(predictions_class) == self.num_layers + 1
        out = {
            "global_embed": output,  # (2, 768)
            # "pred_bins": predictions_class[-1],  # (2, num_queries, 4)
            # "pred_depths": predictions_depth[
            #     -1
            # ],  # [2, num_queries, num_frames, 14, 14]
            # "aux_outputs": self._set_aux_loss(
            #     predictions_class if self.mask_classification else None,
            #     predictions_depth,
            # ),  # outputs['aux_outputs'][0-9]['pred_logits']: torch.Size([2, 100, 3])
            # # outputs['aux_outputs'][0-9]['pred_masks']: torch.Size([2, 100, 4, 14, 14])
        }
        return out

    def get_depth(self, masks, pred_bin):
        """input:
            masks: BS, K, T, H, W
            bins: BS, K, 1
        returns:
            centers: BS, T, H, W
        """
        pred_bin = F.relu(pred_bin) + 0.1  # 0.1 is eps
        bin_widths_normed = (pred_bin / pred_bin.sum(dim=1, keepdim=True)).squeeze(
            2
        )  # BS, K
        # bin_widths = (self.max_val - self.min_val) * bin_widths_normed  # .shape = N, dim_out
        bin_widths = bin_widths_normed  # .shape = N, dim_out
        bin_widths = F.pad(bin_widths, (1, 0), mode="constant", value=0)
        bin_edges = torch.cumsum(bin_widths, dim=1)
        centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
        n, dout = centers.size()
        centers = centers.view(n, dout, 1, 1, 1)

        masks = F.softmax(masks, dim=1)

        pred_depth = torch.sum(masks * centers, dim=1, keepdim=True)
        return pred_depth.squeeze(1)

    def forward_prediction_heads(self, outputs, mask_features, attn_mask_target_size):
        """output: query; mask_features: (key, value)"""
        if outputs.shape[0] % 2 != 0:
            output = outputs[1:, :, :]  # depth queries
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        pred_bins = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape
        pred_depth = self.get_depth(outputs_mask, pred_bins)
        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(
            outputs_mask.flatten(0, 1),
            size=attn_mask_target_size,
            mode="bilinear",
            align_corners=False,
        ).view(b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        if outputs.shape[0] % 2 != 0:
            # For a binary mask, a True value indicates that the corresponding position is not allowed to attend
            # 创建一个全零 tensor，形状为 (b, 1, t, attn_mask_target_size[0], attn_mask_target_size[1])
            zero_tensor = torch.zeros(
                b, 1, t, attn_mask_target_size[0], attn_mask_target_size[1]
            ).cuda()

            # 在 dim=1 维度上拼接
            attn_mask = torch.cat((zero_tensor, attn_mask), dim=1)

        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (
            attn_mask.sigmoid()
            .flatten(2)
            .unsqueeze(1)
            .repeat(1, self.num_heads, 1, 1)
            .flatten(0, 1)
            < 0.5
        ).bool()
        attn_mask = attn_mask.detach()

        # return outputs_class, outputs_mask, attn_mask
        return pred_bins, pred_depth, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_bins": a, "pred_depths": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_depths": b} for b in outputs_seg_masks[:-1]]
