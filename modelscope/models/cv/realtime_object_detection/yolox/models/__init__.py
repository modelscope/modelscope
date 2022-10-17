# The implementation is based on YOLOX, available at https://github.com/Megvii-BaseDetection/YOLOX

from .darknet import CSPDarknet, Darknet
from .dfp_pafpn import DFPPAFPN
from .streamyolo import StreamYOLO
from .tal_head import TALHead
from .yolo_fpn import YOLOFPN
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX
