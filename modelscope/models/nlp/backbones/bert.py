from modelscope.metainfo import Models
from modelscope.models.builder import BACKBONES
from modelscope.models.nlp.bert import BertModel
from modelscope.utils.constant import Fields

BACKBONES.register_module(
    group_key=Fields.nlp, module_name=Models.bert, module_cls=BertModel)
