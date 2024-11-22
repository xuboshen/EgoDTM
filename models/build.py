from .simple_model_wo_shortcut import InstanceHOI as InstanceHOI_wo_shortcut


def build_model(cfg):
    # pretraining models
    if cfg.MODEL.MODEL_NAME == "hoi":
        model = InstanceHOI_wo_shortcut(cfg)
    else:
        raise NotImplementedError(f"{cfg.MODEL.MODEL_NAME} not implemented yet")
    return model
