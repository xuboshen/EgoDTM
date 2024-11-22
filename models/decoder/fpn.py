import math

import torch
import torch.nn as nn
from einops import rearrange

from .modules import Conv2d, _get_activation_fn
from .utils import _get_clones, c2_xavier_fill, get_norm


class SimpleFeaturePyramid(nn.Module):
    """
    This module implements SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        # in_feature,
        out_channels=256,
        scale_factors=(8.0, 4.0, 2.0, 1.0),
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
            out_channels=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5),
            top_block=L(LastLevelMaxPool)(),
            norm="LN",
            square_pad=1024,
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        # input_shapes = net.output_shape()
        stride_default = 16  # 1/16
        strides = [int(stride_default / scale) for scale in scale_factors]
        # _assert_strides_are_log2_contiguous(strides)

        # dim = input_shapes[in_feature].channels
        dim = out_channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 8.0:
                # original fpn
                # layers = [
                #     nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                #     get_norm(norm, dim // 2),
                #     nn.GELU(),
                #     nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                #     get_norm(norm, dim // 4),
                #     nn.GELU(),
                #     nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                # ]
                layers = [
                    nn.ConvTranspose2d(dim, dim // 4, kernel_size=4, stride=4),
                    # get_norm(norm, dim // 4),
                    # nn.GELU(),
                    # nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                    # get_norm(norm, dim // 4),
                    # nn.GELU(),
                    # nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 16.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                    get_norm(norm, dim // 4),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
                    get_norm(norm, dim // 8),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 8, dim // 16, kernel_size=2, stride=2),
                ]
                out_dim = dim // 16
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        # self.net = net
        # self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, features):
        """
        Args:
            feature: Tensor of shape shape(2, 4, 196, 768), (B, T, H*W, Dim)

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        # features = feature[self.in_feature]
        features = rearrange(
            features,
            "b t (h w) d -> (b t) d h w",
            h=int(features.shape[-2] ** 0.5),
            w=int(features.shape[-2] ** 0.5),
        )
        results = []

        for stage in self.stages:
            results.append(stage(features))

        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}
