# The implementation is adopted from VitAdapter,
# made publicly available under the Apache License at https://github.com/czczup/ViT-Adapter.git
from .base import BASEBEiT
from .beit_adapter import BEiTAdapter

__all__ = ['BEiTAdapter', 'BASEBEiT']
