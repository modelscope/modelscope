import uuid
import logging
import os.path as osp
from argparse import Namespace
# from tensorboardX import SummaryWriter

class Base:
    """
    Base configure file, which contains the basic training parameters and should be inherited by other attribute configure file.
    """

    def __init__(self, config_name, ckpt_dir='./', image_dir='./', annot_dir='./'):
        self.type = config_name
        self.id = str(uuid.uuid4())
        self.note = ""

        self.ckpt_dir = ckpt_dir
        self.image_dir = image_dir
        self.annot_dir = annot_dir

        self.loader_type = "alignment"
        self.loss_func = "STARLoss"

        # train
        self.batch_size = 128
        self.val_batch_size = 1
        self.test_batch_size = 32
        self.channels = 3
        self.width = 256
        self.height = 256

        # mean values in r, g, b channel.
        self.means = (127, 127, 127)
        self.scale = 0.0078125

        self.display_iteration = 100
        self.milestones = [50, 80]
        self.max_epoch = 100

        self.net = "stackedHGnet_v1"
        self.nstack = 4

        # ["adam", "sgd"]
        self.optimizer = "adam"
        self.learn_rate = 0.1
        self.momentum = 0.01  # caffe: 0.99
        self.weight_decay = 0.0
        self.nesterov = False
        self.scheduler = "MultiStepLR"
        self.gamma = 0.1

        self.loss_weights = [1.0]
        self.criterions = ["SoftmaxWithLoss"]
        self.metrics = ["Accuracy"]
        self.key_metric_index = 0
        self.classes_num = [1000]
        self.label_num = len(self.classes_num)

        # model
        self.ema = False
        self.use_AAM = True

        # visualization
        self.writer = None

        # log file
        self.logger = None

    def init_instance(self):
        # self.writer = SummaryWriter(logdir=self.log_dir, comment=self.type)
        log_formatter = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
        root_logger = logging.getLogger()
        file_handler = logging.FileHandler(osp.join(self.log_dir, "log.txt"))
        file_handler.setFormatter(log_formatter)
        file_handler.setLevel(logging.NOTSET)
        root_logger.addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        console_handler.setLevel(logging.NOTSET)
        root_logger.addHandler(console_handler)
        root_logger.setLevel(logging.NOTSET)
        self.logger = root_logger

    def __del__(self):
        # tensorboard --logdir self.log_dir
        if self.writer is not None:
            # self.writer.export_scalars_to_json(self.log_dir + "visual.json")
            self.writer.close()

    def init_from_args(self, args: Namespace):
        args_vars = vars(args)
        for key, value in args_vars.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
