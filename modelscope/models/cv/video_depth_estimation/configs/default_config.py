# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
"""Default dro_sfm configuration parameters (overridable in configs/*.yaml)
"""

import os

from yacs.config import CfgNode as CN

########################################################################################################################
cfg = CN()
cfg.name = ''  # Run name
cfg.debug = False  # Debugging flag
########################################################################################################################
# ARCH
########################################################################################################################
cfg.arch = CN()
cfg.arch.seed = 42  # Random seed for Pytorch/Numpy initialization
cfg.arch.min_epochs = 1  # Minimum number of epochs
cfg.arch.max_epochs = 50  # Maximum number of epochs
########################################################################################################################
# CHECKPOINT
########################################################################################################################
cfg.checkpoint = CN()
cfg.checkpoint.filepath = './results/mdoel'  # Checkpoint filepath to save data
cfg.checkpoint.save_top_k = 5  # Number of best models to save
cfg.checkpoint.monitor = 'abs_rel_pp_gt'  # Metric to monitor for logging
cfg.checkpoint.monitor_index = 0  # Dataset index for the metric to monitor
cfg.checkpoint.mode = 'auto'  # Automatically determine direction of improvement (increase or decrease)
cfg.checkpoint.s3_path = ''  # s3 path for AWS model syncing
cfg.checkpoint.s3_frequency = 1  # How often to s3 sync
########################################################################################################################
# SAVE
########################################################################################################################
cfg.save = CN()
cfg.save.folder = './results'  # Folder where data will be saved
cfg.save.depth = CN()
cfg.save.depth.rgb = True  # Flag for saving rgb images
cfg.save.depth.viz = True  # Flag for saving inverse depth map visualization
cfg.save.depth.npz = True  # Flag for saving numpy depth maps
cfg.save.depth.png = True  # Flag for saving png depth maps
########################################################################################################################
# WANDB
########################################################################################################################
cfg.wandb = CN()
cfg.wandb.dry_run = True  # Wandb dry-run (not logging)
cfg.wandb.name = ''  # Wandb run name
cfg.wandb.project = os.environ.get('WANDB_PROJECT', '')  # Wandb project
cfg.wandb.entity = os.environ.get('WANDB_ENTITY', '')  # Wandb entity
cfg.wandb.tags = []  # Wandb tags
cfg.wandb.dir = ''  # Wandb save folder
########################################################################################################################
# MODEL
########################################################################################################################
cfg.model = CN()
cfg.model.name = ''  # Training model
cfg.model.checkpoint_path = ''  # Checkpoint path for model saving
########################################################################################################################
# MODEL.OPTIMIZER
########################################################################################################################
cfg.model.optimizer = CN()
cfg.model.optimizer.name = 'Adam'  # Optimizer name
cfg.model.optimizer.depth = CN()
cfg.model.optimizer.depth.lr = 0.0002  # Depth learning rate
cfg.model.optimizer.depth.weight_decay = 0.0  # Dept weight decay
cfg.model.optimizer.pose = CN()
cfg.model.optimizer.pose.lr = 0.0002  # Pose learning rate
cfg.model.optimizer.pose.weight_decay = 0.0  # Pose weight decay
cfg.model.optimizer.momentum = 0.9
########################################################################################################################
# MODEL.SCHEDULER
########################################################################################################################
cfg.model.scheduler = CN()
cfg.model.scheduler.name = 'StepLR'  # Scheduler name
cfg.model.scheduler.step_size = 10  # Scheduler step size
cfg.model.scheduler.gamma = 0.5  # Scheduler gamma value
cfg.model.scheduler.T_max = 20  # Scheduler maximum number of iterations
cfg.model.scheduler.eta_min = 1e-7
cfg.model.scheduler.milestones = [10, 15, 20, 25, 30, 35, 40, 45]
########################################################################################################################
# MODEL.PARAMS
########################################################################################################################
cfg.model.params = CN()
cfg.model.params.crop = ''  # Which crop should be used during evaluation
cfg.model.params.min_depth = 0.0  # Minimum depth value to evaluate
cfg.model.params.max_depth = 80.0  # Maximum depth value to evaluate
########################################################################################################################
# MODEL.LOSS
########################################################################################################################
cfg.model.loss = CN()
#
cfg.model.loss.num_scales = 4  # Number of inverse depth scales to use
cfg.model.loss.progressive_scaling = 0.0  # Training percentage to decay number of scales
cfg.model.loss.flip_lr_prob = 0.5  # Probablity of horizontal flippping
cfg.model.loss.rotation_mode = 'euler'  # Rotation mode
cfg.model.loss.upsample_depth_maps = True  # Resize depth maps to highest resolution
#
cfg.model.loss.ssim_loss_weight = 0.85  # SSIM loss weight
cfg.model.loss.occ_reg_weight = 0.1  # Occlusion regularizer loss weight
cfg.model.loss.smooth_loss_weight = 0.001  # Smoothness loss weight
cfg.model.loss.C1 = 1e-4  # SSIM parameter
cfg.model.loss.C2 = 9e-4  # SSIM parameter
cfg.model.loss.photometric_reduce_op = 'min'  # Method for photometric loss reducing
cfg.model.loss.disp_norm = True  # Inverse depth normalization
cfg.model.loss.clip_loss = 0.0  # Clip loss threshold variance
cfg.model.loss.padding_mode = 'zeros'  # Photometric loss padding mode
cfg.model.loss.automask_loss = True  # Automasking to remove static pixels
#
cfg.model.loss.velocity_loss_weight = 0.1  # Velocity supervision loss weight
#
cfg.model.loss.supervised_method = 'sparse-l1'  # Method for depth supervision
cfg.model.loss.supervised_num_scales = 4  # Number of scales for supervised learning
cfg.model.loss.supervised_loss_weight = 0.9  # Supervised loss weight
########################################################################################################################
# MODEL.DEPTH_NET
########################################################################################################################
cfg.model.depth_net = CN()
cfg.model.depth_net.name = ''  # Depth network name
cfg.model.depth_net.checkpoint_path = ''  # Depth checkpoint filepath
cfg.model.depth_net.version = ''  # Depth network version
cfg.model.depth_net.dropout = 0.0  # Depth network dropout
########################################################################################################################
# MODEL.POSE_NET
########################################################################################################################
cfg.model.pose_net = CN()
cfg.model.pose_net.name = ''  # Pose network name
cfg.model.pose_net.checkpoint_path = ''  # Pose checkpoint filepath
cfg.model.pose_net.version = ''  # Pose network version
cfg.model.pose_net.dropout = 0.0  # Pose network dropout
########################################################################################################################
# MODEL.perpcep_net
########################################################################################################################
cfg.model.percep_net = CN()
cfg.model.percep_net.name = ''  # percep_net network name
cfg.model.percep_net.checkpoint_path = ''  # percep_net checkpoint filepath
cfg.model.percep_net.version = ''  # percep_net network version
cfg.model.percep_net.dropout = 0.0  # percep_net network dropout
########################################################################################################################
# DATASETS
########################################################################################################################
cfg.datasets = CN()
########################################################################################################################
# DATASETS.AUGMENTATION
########################################################################################################################
cfg.datasets.augmentation = CN()
cfg.datasets.augmentation.image_shape = (192, 640)  # Image shape
cfg.datasets.augmentation.jittering = (0.2, 0.2, 0.2, 0.05
                                       )  # Color jittering values
########################################################################################################################
# DATASETS.TRAIN
########################################################################################################################
cfg.datasets.train = CN()
cfg.datasets.train.batch_size = 8  # Training batch size
cfg.datasets.train.num_workers = 16  # Training number of workers
cfg.datasets.train.back_context = 1  # Training backward context
cfg.datasets.train.forward_context = 1  # Training forward context
cfg.datasets.train.dataset = []  # Training dataset
cfg.datasets.train.path = []  # Training data path
cfg.datasets.train.split = []  # Training split
cfg.datasets.train.depth_type = ['']  # Training depth type
cfg.datasets.train.cameras = [
    []
]  # Training cameras (double list, one for each dataset)
cfg.datasets.train.repeat = [
    1
]  # Number of times training dataset is repeated per epoch
cfg.datasets.train.num_logs = 5  # Number of training images to log
cfg.datasets.train.strides = (1, )  # stride
########################################################################################################################
# DATASETS.VALIDATION
########################################################################################################################
cfg.datasets.validation = CN()
cfg.datasets.validation.batch_size = 1  # Validation batch size
cfg.datasets.validation.num_workers = 8  # Validation number of workers
cfg.datasets.validation.back_context = 0  # Validation backward context
cfg.datasets.validation.forward_context = 0  # Validation forward contxt
cfg.datasets.validation.dataset = []  # Validation dataset
cfg.datasets.validation.path = []  # Validation data path
cfg.datasets.validation.split = []  # Validation split
cfg.datasets.validation.depth_type = ['']  # Validation depth type
cfg.datasets.validation.cameras = [
    []
]  # Validation cameras (double list, one for each dataset)
cfg.datasets.validation.num_logs = 5  # Number of validation images to log
cfg.datasets.validation.strides = (1, )  # stride
########################################################################################################################
# DATASETS.TEST
########################################################################################################################
cfg.datasets.test = CN()
cfg.datasets.test.batch_size = 1  # Test batch size
cfg.datasets.test.num_workers = 8  # Test number of workers
cfg.datasets.test.back_context = 0  # Test backward context
cfg.datasets.test.forward_context = 0  # Test forward context
cfg.datasets.test.dataset = []  # Test dataset
cfg.datasets.test.path = []  # Test data path
cfg.datasets.test.split = []  # Test split
cfg.datasets.test.depth_type = ['']  # Test depth type
cfg.datasets.test.cameras = [
    []
]  # Test cameras (double list, one for each dataset)
cfg.datasets.test.num_logs = 5  # Number of test images to log
cfg.datasets.test.strides = (1, )  # stride
########################################################################################################################
# THESE SHOULD NOT BE CHANGED
########################################################################################################################
cfg.config = ''  # Run configuration file
cfg.default = ''  # Run default configuration file
cfg.wandb.url = ''  # Wandb URL
cfg.checkpoint.s3_url = ''  # s3 URL
cfg.save.pretrained = ''  # Pretrained checkpoint
cfg.prepared = False  # Prepared flag
########################################################################################################################


def get_cfg_defaults():
    return cfg.clone()
