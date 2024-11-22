from .model_clip import CLIP_VITB16


def build_avion(cfg, **kwargs):
    if cfg.MODEL.AVION.MODEL_TYPE == "CLIP_VITB16":
        model = CLIP_VITB16(
            freeze_temperature=cfg.MODEL.AVION.FREEZE_TEMPERATURE,
            use_grad_checkpointing=cfg.MODEL.AVION.USE_GRADIENT_CHECKPOINTING,
            use_bidirectional_lm=cfg.MODEL.AVION.FREEZE_TEMPERATURE,
            context_length=cfg.MODEL.AVION.TEXT.CONTEXT_LEN,
            patch_dropout=0.0,
            drop_path_rate=0.0,
            num_frames=cfg.INPUT.NUM_FRAMES,
            use_fast_conv1=cfg.MODEL.AVION.USE_FAST_CONV1,
            use_flash_attn=cfg.MODEL.AVION.USE_FLASH_ATTN,
            project_embed_dim=cfg.MODEL.AVION.PROJECT_EMBED_DIM,
            pretrain_zoo=cfg.MODEL.AVION.PRETRAIN_ZOO,
            pretrain_path=None,
            cfg=cfg,
            **kwargs,
        )
    else:
        raise (f"{cfg.MODEL.AVION.MODEL_TYPE} Not Implemented Error")
    return model
