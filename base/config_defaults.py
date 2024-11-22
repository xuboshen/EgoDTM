from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.LOSS = CN()
_C.LOSS.NAME = "infonce"
_C.LOSS.DEPTH = CN()
_C.LOSS.DEPTH.PARAM_CHAMFER = 0.1


_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.MODEL_NAME = "hoi"
_C.MODEL.BACKBONE = "avion"  # use it if MODEL_NAME == 'hoi
_C.MODEL.USE_CAP = False
_C.MODEL.WITH_PIXEL_DECODER = False
_C.MODEL.ONLY_FPN = True
_C.MODEL.WO_SHORTCUT = True
_C.MODEL.DEBUG = False
_C.MODEL.LARGE_MAP = False
_C.MODEL.USE_VIDEOECLS = True

# -----------------------------------------------------------------------------
# MASK2FORMER configurations
# -----------------------------------------------------------------------------
_C.MODEL.MASK2FORMER = CN()
_C.MODEL.MASK2FORMER.DEC_TYPE = "w_pixel_d"
_C.MODEL.MASK2FORMER.USE_FLASH_ATTN = True
# _C.MODEL.MASK2FORMER.NUM_CLASSES = 256
_C.MODEL.MASK2FORMER.HIDDEN_DIM = 256
_C.MODEL.MASK2FORMER.NUM_OBJECT_QUERIES = 8
_C.MODEL.MASK2FORMER.VIDEO_QUERIES = 0
_C.MODEL.MASK2FORMER.NUM_HEADS = 8
_C.MODEL.MASK2FORMER.DIM_FEEDFORWARD = 2048
_C.MODEL.MASK2FORMER.DEC_LAYERS = 6
_C.MODEL.MASK2FORMER.PRE_NORM = False
_C.MODEL.MASK2FORMER.ENFORCE_INPUT_PROJ = False
_C.MODEL.MASK2FORMER.MASK_DIM = 256
_C.MODEL.MASK2FORMER.PIXEL_MEAN = [123.675, 116.280, 103.530]
_C.MODEL.MASK2FORMER.PIXEL_STD = [58.395, 57.120, 57.375]

# hand and object
_C.MODEL.MASK2FORMER.NUM_CLASSES = 16

# in models/models
_C.MODEL.MASK2FORMER.DEEP_SUPERVISION = True
_C.MODEL.MASK2FORMER.NO_OBJECT_WEIGHT = 0.1
_C.MODEL.MASK2FORMER.CLASS_WEIGHT = 2.0
_C.MODEL.MASK2FORMER.DICE_WEIGHT = 5.0
_C.MODEL.MASK2FORMER.MASK_WEIGHT = 5.0
_C.MODEL.MASK2FORMER.TRAIN_NUM_POINTS = 12544
_C.MODEL.MASK2FORMER.OVERSAMPLE_RATIO = 3.0
_C.MODEL.MASK2FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
_C.MODEL.MASK2FORMER.SIZE_DIVISIBILITY = 32
# For pixel decoders
_C.MODEL.MASK2FORMER.CONVS_DIM = 256
_C.MODEL.MASK2FORMER.NORM = "GN"
_C.MODEL.MASK2FORMER.DROPOUT = 0.0
_C.MODEL.MASK2FORMER.TRANSFORMER_ENC_LAYERS = 6
_C.MODEL.MASK2FORMER.COMMON_STRIDE = 4
# tmp parameters
_C.MODEL.MASK2FORMER.IN_FEATURES = ["p2", "p4"]
_C.MODEL.MASK2FORMER.NUM_FEATURE_LEVELS = 1
_C.MODEL.MASK2FORMER.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["p4"]


_C.MODEL.MASK2FORMER.TEST = CN()
_C.MODEL.MASK2FORMER.TEST.OVERLAP_THRESHOLD = 0.8
_C.MODEL.MASK2FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8
_C.MODEL.MASK2FORMER.TEST.SIZE_DIVISIBILITY = 32
_C.MODEL.MASK2FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8
_C.MODEL.MASK2FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.8

# -----------------------------------------------------------------------------
# LAVILA configurations
# -----------------------------------------------------------------------------
_C.MODEL.LAVILA = CN()
_C.MODEL.LAVILA.TEMPERATURE_INIT = 0.07
# projection space dim of video-text alignment
_C.MODEL.LAVILA.PROJECT_EMBED_DIM = 256
# freeze the text encoder
_C.MODEL.LAVILA.FREEZE_TEXT_BACKBONE = False
# freeze the clip pretrained parts
_C.MODEL.LAVILA.FREEZE_CLIP = False
# normalize the embedding or not
_C.MODEL.LAVILA.NORM_EMBED = True
# Turn it on if using model 'hoi'
_C.MODEL.LAVILA.RETURN_FEAT = True

# lora params
_C.USE_LORA = False
_C.LORA_CONFIG = CN()
_C.LORA_CONFIG.RANK = 16
_C.LORA_CONFIG.ALPHA = 16
_C.LORA_CONFIG.DROPOUT = 0
_C.LORA_CONFIG.PARAMS = ["timeattn", "attn", "mlp"]

# LaViLa visual encoder
_C.MODEL.LAVILA.TIMESFORMER = CN()
_C.MODEL.LAVILA.TIMESFORMER.VIT_SIZE = "ViT-B/16"
_C.MODEL.LAVILA.TIMESFORMER.IMG_SIZE = 224
_C.MODEL.LAVILA.TIMESFORMER.PATCH_SIZE = 16
# hidden dim
_C.MODEL.LAVILA.TIMESFORMER.EMBED_DIM = 768
# num of layers
_C.MODEL.LAVILA.TIMESFORMER.DEPTH = 12
_C.MODEL.LAVILA.TIMESFORMER.NUM_HEADS = 12
_C.MODEL.LAVILA.TIMESFORMER.TIME_INIT = "zeros"
_C.MODEL.LAVILA.TIMESFORMER.ATTENTION_STYLE = "frozen-in-time"
# prenorm or postnorm
_C.MODEL.LAVILA.TIMESFORMER.LN_PRE = True
_C.MODEL.LAVILA.TIMESFORMER.DROP_PATH_RATE = 0
_C.MODEL.LAVILA.TIMESFORMER.TIMESFORMER_GATED_XATTN = False

# lavila text encoder
_C.MODEL.LAVILA.TEXT = CN()
_C.MODEL.LAVILA.TEXT.EMBED_DIM = 512
_C.MODEL.LAVILA.TEXT.NUM_HEADS = 8
_C.MODEL.LAVILA.TEXT.DEPTH = 12


# -----------------------------------------------------------------------------
# AVION configurations
# -----------------------------------------------------------------------------
_C.MODEL.AVION = CN()
_C.MODEL.AVION.MODEL_TYPE = "CLIP_VITB16"

_C.MODEL.AVION.FREEZE_TEMPERATURE = True
# projection space dim of video-text alignment
_C.MODEL.AVION.USE_GRADIENT_CHECKPOINTING = True
_C.MODEL.AVION.USE_FAST_CONV1 = True
_C.MODEL.AVION.USE_FLASH_ATTN = True
_C.MODEL.AVION.PROJECT_EMBED_DIM = 256
_C.MODEL.AVION.PRETRAIN_ZOO = "openai"  # choose from 'lavila', 'open_clip', 'openai'
# Turn it on if using model 'hoi'
_C.MODEL.AVION.RETURN_FEAT = False
_C.MODEL.AVION.RETURN_LOW_FEAT = False

# freeze the text encoder
_C.MODEL.AVION.TEXT = CN()
_C.MODEL.AVION.TEXT.CONTEXT_LEN = 77


# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = 32
# Size of the image during test
_C.INPUT.SIZE_TEST = 32
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [
    0.1307,
]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [
    0.3081,
]
_C.INPUT.NUM_FRAMES = 4


# -----------------------------------------------------------------------------
# Task
# -----------------------------------------------------------------------------
_C.TASK = CN()
# VTC: video-text contrastive, DepthEstim: DepthEstimation
_C.TASK.TASK_NAME = ["VTC", "DepthEstim"]

# -----------------------------------------------------------------------------
# Downstream EgoNLQ Task, everything about the parameters of EgoNLQ
# -----------------------------------------------------------------------------
_C.TASK.EGONLQ = CN()
_C.TASK.EGONLQ.EXAMPLE_PARAM = 0

_C.TASK.OSCC = CN()
_C.TASK.OSCC.EXAMPLE_PARAM = 0
# -----------------------------------------------------------------------------
# Dataset
# Dataset_name choose from: ['VideoCaptionEgo4D', 'EgoNLQ', 'EgoMQ']
# -----------------------------------------------------------------------------
_C.DATASET = CN()

# List of the dataset names for training, as present in paths_catalog.py
_C.DATASET.TRAIN = CN()
_C.DATASET.TRAIN.DATASET_NAME = "VideoCaptionEgo4D"
_C.DATASET.TRAIN.ANNO_PATH = ""
_C.DATASET.TRAIN.DATA_PATH = ""
_C.DATASET.TRAIN.DEPTH_PATH = ""
_C.DATASET.TRAIN.METADATA_AUX = None
_C.DATASET.TRAIN.FUSED_DECODE_CROP = True
_C.DATASET.TRAIN.CHUNK_LEN = 300  # by default, faster: 15
_C.DATASET.TRAIN.AUGMENTED_TEXT = False

# List of the dataset names for evaluation
# Dataset_name choose from: ['V2TEgoHOIEgo4D', 'EgoMQ', 'EK100MIR']
_C.DATASET.VAL = CN()
_C.DATASET.VAL.FUSED_DECODE_CROP = [True]
# _C.DATASET.VAL.DATASET_NAME = "V2TEgoHOIEgo4D"
_C.DATASET.VAL.ANNO_PATH = [""]
_C.DATASET.VAL.DATA_PATH = [""]
_C.DATASET.VAL.CHUNK_LEN = [15]  # by default, faster: 15

_C.DATASET.VAL.DATASET_NAMES = ["ek100mir"]  # ek100mir, egomcq
_C.DATASET.VAL.EK100MIR = CN()
_C.DATASET.VAL.EK100MIR.METADATA = "EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv"
_C.DATASET.VAL.EK100MIR.RELEVANCY_PATH = "EK100_320p_15sec_30fps_libx264/epic-kitchens-100-annotations/caption_relevancy_EPIC_100_retrieval_test.pkl"


# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASET.TEST = CN()
_C.DATASET.TEST.DATASET_NAME = "VideoCaptionEgo4D"
_C.DATASET.TEST.ANNO_PATH = ""
_C.DATASET.TEST.DATA_PATH = ""

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.GRADIENT_CHECKPOINT = True

_C.SOLVER.OPTIMIZER_NAME = "AdamW"
_C.SOLVER.BASE_LR = 3e-5
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.WEIGHT_DECAY = 0.01
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 100
_C.SOLVER.MIXED_PRECISION = "no"  # select from ["bf16", "fp16", "no"]


_C.SOLVER.SGD = CN()
_C.SOLVER.SGD.MOMENTUM = 0.9
_C.SOLVER.SGD.GAMMA = 0.1

_C.SOLVER.ADAMW = CN()
_C.SOLVER.ADAMW.BETAS = (0.9, 0.999)
_C.SOLVER.ADAMW.EPS = 1e-8


# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.BATCH_SIZE = 16

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "test_results"
_C.CKPT_DIR = "egodtm_best_vtc_depth_augtext.pt"
_C.RESUME = ""
_C.PRINT_FREQ = 10
_C.USE_WANDB = False
_C.PRE_EVAL = True
_C.EVAL_FREQ = 1


# for fine-tune settings only
_C.DOWNSTREAM = CN()
_C.DOWNSTREAM.DATASETS = ["ek100mir"]
_C.DOWNSTREAM.EK100MIR = CN()
_C.DOWNSTREAM.EK100MIR.VAL = CN()
_C.DOWNSTREAM.EK100MIR.VAL.METADATA = "EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/EPIC_100_retrieval_test.csv"
_C.DOWNSTREAM.EK100MIR.VAL.RELEVANCY_PATH = "EK100_256p/epic-kitchens-100-annotations/retrieval_annotations/relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl"
