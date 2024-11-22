from .decoder_w_pixel_d import MaskedTransformerDecoder as Decoder_with_PixelDecoder
from .simple_decoder_w_pixel_d import (
    MaskedTransformerDecoder as SimpleDecoder_with_PixelDecoder,
)
def build_decoder(cfg, in_channel, dtype="w_pixel_d"):
    if dtype == "w_pixel_d":
        model = Decoder_with_PixelDecoder(cfg=cfg, in_channels=in_channel)
    elif dtype == "simple_w_pixel_d":
        model = SimpleDecoder_with_PixelDecoder(cfg=cfg, in_channels=in_channel)
    return model
