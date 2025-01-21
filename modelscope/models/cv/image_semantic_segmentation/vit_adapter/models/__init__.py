# The implementation is adopted from VitAdapter,
# made publicly available under the Apache License at https://github.com/czczup/ViT-Adapter.git
from .backbone import BASEBEiT, BEiTAdapter
from .decode_heads import Mask2FormerHeadFromMMSeg
from .segmentors import EncoderDecoderMask2Former
