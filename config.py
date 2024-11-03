"""
Default configurations for action recognition domain adaptation
"""

import os

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.ROOT = "../CTAN-main/data/"  # "/shared/tale2/Shared"
_C.DATASET.SOURCE = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.SRC_TRAINLIST = "epic_D2_train.pkl"
_C.DATASET.SRC_TESTLIST = "epic_D2_test.pkl"
_C.DATASET.TARGET = "EPIC"  # dataset options=["EPIC", "GTEA", "ADL", "KITCHEN"]
_C.DATASET.TGT_TRAINLIST = "epic_D3_train.pkl"
_C.DATASET.TGT_TESTLIST = "epic_D3_test.pkl"
_C.DATASET.IMAGE_MODALITY = "rgb"  # mode options=["rgb", "flow", "joint"]
# _C.DATASET.NUM_CLASSES = 8
_C.DATASET.FRAMES_PER_SEGMENT = 16
_C.DATASET.NUM_REPEAT = 1  # 10
_C.DATASET.WEIGHT_TYPE = "natural"
_C.DATASET.SIZE_TYPE = "max"  # options=["source", "max"]
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2020
_C.SOLVER.BASE_LR = 0.0001  # Initial learning rate
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 0.0005  # 1e-4
_C.SOLVER.NESTEROV = True

_C.SOLVER.TYPE = "SGD"
_C.SOLVER.MAX_EPOCHS = 50  # "nb_adapt_epochs": 100,
# _C.SOLVER.WARMUP = True
_C.SOLVER.MIN_EPOCHS = 20  # "nb_init_epochs": 20,
_C.SOLVER.TRAIN_BATCH_SIZE = 6
# _C.SOLVER.TEST_BATCH_SIZE = 32  # No difference in ADA
#UNIFORMER
_C.UNIFORMER = CN()
_C.UNIFORMER.DEPTH = [3, 4, 8, 3]
_C.UNIFORMER.HEAD_DIM = 64
_C.UNIFORMER.MLP_RATIO = 4
_C.UNIFORMER.DROPOUT_RATE = 0
_C.UNIFORMER.ATTENTION_DROPOUT_RATE = 0
_C.UNIFORMER.DROP_DEPTH_RATE = 0.1
_C.UNIFORMER.SPLIT = False
_C.UNIFORMER.QKV_BIAS = True
_C.UNIFORMER.QKV_SCALE = None
_C.UNIFORMER.PRETRAIN_NAME = 'uniformer_small_k400_8x8'
_C.UNIFORMER.EMBED_DIM = [64, 128, 320, 512]  # [64, 128, 320, 512]
# depth.
_C.UNIFORMER.DEPTH = [3, 4, 8, 3]
# dimension of head.
_C.UNIFORMER.HEAD_DIM = 64
# ratio of mlp hidden dim to embedding dim.
_C.UNIFORMER.MLP_RATIO = 4
# enable bias for qkv if True.
_C.UNIFORMER.QKV_BIAS = True
# override default qk scale of head_dim ** -0.5 if set.
_C.UNIFORMER.QKV_SCALE = None
# enable and set representation layer (pre-logits) to this value if set.
_C.UNIFORMER.REPRESENTATION_SIZE = None
# dropout rate.
_C.UNIFORMER.DROPOUT_RATE = 0
# attention dropout rate.
_C.UNIFORMER.ATTENTION_DROPOUT_RATE = 0
# stochastic depth rate.
_C.UNIFORMER.DROP_DEPTH_RATE = 0.1
# pretrained name.

# whether use split attention.
_C.UNIFORMER.SPLIT = False
# type stage.
_C.UNIFORMER.STAGE_TYPE = [0, 0, 1, 1]
# spatial-temporal downsample.
_C.UNIFORMER.STD = False
# prune ratio.
_C.UNIFORMER.PRUNE_RATIO = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
# trade off.
_C.UNIFORMER.TRADE_OFF = [[], [], [1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]
# Adaptation-specific solver config
_C.SOLVER.AD_LAMBDA = True
_C.SOLVER.AD_LR = True
_C.SOLVER.INIT_LAMBDA = 1.0

#DATA:
_C.DATA = CN()
_C.DATA.USE_OFFSET_SAMPLING = True
# _C.DATA.DECODING_BACKEND = decord
_C.DATA.NUM_FRAMES = 8
_C.DATA.SAMPLING_RATE = 8
_C.DATA.TRAIN_JITTER_SCALES = [224, 224]
_C.DATA.TRAIN_CROP_SIZE = 224
_C.DATA.TEST_CROP_SIZE = 224
_C.DATA.INPUT_CHANNEL_NUM = [3]
_C.DATA.PATH_TO_DATA_DIR = '/data/ZouYiShan/EpicData_rgb/label/'
_C.DATA.TRAIN_JITTER_SCALES_RELATIVE = [0.08, 1.0]
_C.DATA.TRAIN_JITTER_ASPECT_RELATIVE = [0.75, 1.3333]

# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.MODEL = CN()
_C.MODEL.METHOD = "i3d"  # options=["r3d_18", "r2plus1d_18", "mc3_18", "i3d"]
_C.MODEL.ATTENTION = "None"  # options=["None", "SELayer", "SELayerC", "SELayerCT", "SELayerTC"]
_C.MODEL.NUM_CLASSES_V = 8
_C.MODEL.NUM_CLASSES_N = 42
_C.MODEL.USE_CHECKPOINT = False
_C.MODEL.CHECKPOINT_NUM = [0, 0, 0, 0]
_C.MODEL.ARCH = 'uniformer'
# ---------------------------------------------------------------------------- #
# Domain Adaptation Net (DAN) configs
# ---------------------------------------------------------------------------- #
_C.DAN = CN()
_C.DAN.METHOD = "CDAN"  # options=["CDAN", "CDAN-E", "DANN", "DAN"]
_C.DAN.USERANDOM = False
_C.DAN.RANDOM_DIM = 1024
# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.VERBOSE = False  # To discuss, for HPC jobs
_C.OUTPUT.FAST_DEV_RUN = False  # True for debug
_C.OUTPUT.PB_FRESH = 0  # 0 # 50 # 0 to disable  ; MAYBE make it a command line option
_C.OUTPUT.TB_DIR = os.path.join("lightning_logs", _C.DATASET.SOURCE + "2" + _C.DATASET.TARGET)


def get_cfg_defaults():
    return _C.clone()
