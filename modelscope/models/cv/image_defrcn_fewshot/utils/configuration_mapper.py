# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/config/defaults.py

from detectron2.config.defaults import _C

from modelscope.utils.config import Config


def detectron2_default_cfg():

    _CC = _C

    # ----------- Backbone ----------- #
    _CC.MODEL.BACKBONE.FREEZE = False
    _CC.MODEL.BACKBONE.FREEZE_AT = 3

    # ------------- RPN -------------- #
    _CC.MODEL.RPN.FREEZE = False
    _CC.MODEL.RPN.ENABLE_DECOUPLE = False
    _CC.MODEL.RPN.BACKWARD_SCALE = 1.0

    # ------------- ROI -------------- #
    _CC.MODEL.ROI_HEADS.NAME = 'Res5ROIHeads'
    _CC.MODEL.ROI_HEADS.FREEZE_FEAT = False
    _CC.MODEL.ROI_HEADS.ENABLE_DECOUPLE = False
    _CC.MODEL.ROI_HEADS.BACKWARD_SCALE = 1.0
    _CC.MODEL.ROI_HEADS.OUTPUT_LAYER = 'FastRCNNOutputLayers'
    _CC.MODEL.ROI_HEADS.CLS_DROPOUT = False
    _CC.MODEL.ROI_HEADS.DROPOUT_RATIO = 0.8
    _CC.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7  # for faster

    # ------------- TEST ------------- #
    _CC.TEST.PCB_ENABLE = False
    _CC.TEST.PCB_MODELTYPE = 'resnet'  # res-like
    _CC.TEST.PCB_MODELPATH = ''
    _CC.TEST.PCB_ALPHA = 0.50
    _CC.TEST.PCB_UPPER = 1.0
    _CC.TEST.PCB_LOWER = 0.05

    # ------------ Other ------------- #
    _CC.SOLVER.WEIGHT_DECAY = 5e-5
    _CC.MUTE_HEADER = True

    return _CC


class CfgMapper():

    def __init__(self, cfg: Config):

        self.cfg = cfg
        self.model_cfg = detectron2_default_cfg().clone()

    def __call__(self, *args, **kwargs):
        cfg_list = [
            'MODEL.WEIGHTS',
            self.cfg.safe_get('model.weights', ''), 'MODEL.MASK_ON',
            self.cfg.safe_get('model.mask_on', False), 'MODEL.BACKBONE.FREEZE',
            self.cfg.safe_get('model.backbone.freezed',
                              False), 'MODEL.RESNETS.DEPTH',
            self.cfg.safe_get('model.resnets.depth',
                              101), 'MODEL.ROI_HEADS.ENABLE_DECOUPLE',
            self.cfg.safe_get('model.roi_heads.enable_decouple',
                              False), 'MODEL.ROI_HEADS.BACKWARD_SCALE',
            self.cfg.safe_get('model.roi_heads.backward_scale',
                              1.0), 'MODEL.ROI_HEADS.NUM_CLASSES',
            self.cfg.safe_get('model.roi_heads.num_classes',
                              80), 'MODEL.ROI_HEADS.FREEZE_FEAT',
            self.cfg.safe_get('model.roi_heads.freeze_feat',
                              False), 'MODEL.ROI_HEADS.CLS_DROPOUT',
            self.cfg.safe_get('model.roi_heads.cls_dropout',
                              False), 'MODEL.RPN.ENABLE_DECOUPLE',
            self.cfg.safe_get('model.rpn.enable_decouple',
                              False), 'MODEL.RPN.BACKWARD_SCALE',
            self.cfg.safe_get('model.rpn.backward_scale',
                              1.0), 'MODEL.RPN.FREEZE',
            self.cfg.safe_get('model.rpn.freezed',
                              False), 'MODEL.RPN.PRE_NMS_TOPK_TEST',
            self.cfg.safe_get('model.rpn.pre_nms_topk_test',
                              6000), 'MODEL.RPN.POST_NMS_TOPK_TEST',
            self.cfg.safe_get('model.rpn.post_nms_topk_test',
                              1000), 'DATASETS.TRAIN',
            tuple(self.cfg.safe_get('datasets.train',
                                    ('coco_2017_train', ))), 'DATASETS.TEST',
            tuple(self.cfg.safe_get('datasets.test', ('coco_2017_val', ))),
            'SOLVER.IMS_PER_BATCH',
            self.cfg.safe_get('train.dataloader.ims_per_batch',
                              16), 'SOLVER.BASE_LR',
            self.cfg.safe_get('train.optimizer.lr', 0.02), 'SOLVER.STEPS',
            tuple(
                self.cfg.safe_get('train.lr_scheduler.steps',
                                  (60000, 80000))), 'SOLVER.MAX_ITER',
            self.cfg.safe_get('train.max_iter',
                              90000), 'SOLVER.CHECKPOINT_PERIOD',
            self.cfg.safe_get('train.checkpoint_period',
                              5000), 'SOLVER.WARMUP_ITERS',
            self.cfg.safe_get('train.lr_scheduler.warmup_iters',
                              1000), 'OUTPUT_DIR',
            self.cfg.safe_get('train.work_dir',
                              './output/'), 'INPUT.MIN_SIZE_TRAIN',
            tuple(
                self.cfg.safe_get('input.min_size_train',
                                  (640, 672, 704, 736, 768, 800))),
            'INPUT.MIN_SIZE_TEST',
            self.cfg.safe_get('input.min_size_test', 800), 'TEST.PCB_ENABLE',
            self.cfg.safe_get('test.pcb_enable', False), 'TEST.PCB_MODELPATH',
            self.cfg.safe_get('test.pcb_modelpath', '')
        ]

        self.model_cfg.merge_from_list(cfg_list)

        return self.model_cfg
