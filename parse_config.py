from yacs.config import CfgNode as AttrDict

__C = AttrDict()

__C.MODEL = AttrDict()
__C.MODEL.NUM_WRITERS = 1000  # Set as appropriate
__C.MODEL.WRITER_EMB_DIM = 320

# Remove all style encoder and laplace related configs

__C.SOLVER = AttrDict()
__C.SOLVER.TYPE = 'AdamW'
__C.SOLVER.BASE_LR = 0.001
__C.SOLVER.EPOCHS = 20000
__C.SOLVER.WARMUP_ITERS = 0
