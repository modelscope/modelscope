from transformers import PreTrainedModel

from modelscope.metainfo import Models
from modelscope.models.base import TorchModel
from modelscope.models.builder import BACKBONES
from modelscope.models.nlp.structbert import SbertConfig
from modelscope.models.nlp.structbert import SbertModel as SbertModelTransform
from modelscope.utils.constant import Fields
from modelscope.utils.logger import get_logger

logger = get_logger(__name__)


@BACKBONES.register_module(Fields.nlp, module_name=Models.structbert)
class SbertModel(TorchModel, SbertModelTransform):

    def __init__(self, model_dir=None, add_pooling_layer=True, **config):
        """
        Args:
            model_dir (str, optional): The model checkpoint directory. Defaults to None.
            add_pooling_layer (bool, optional): to decide if pool the output from hidden layer. Defaults to True.
        """
        config = SbertConfig(**config)
        super().__init__(model_dir)
        self.config = config
        SbertModelTransform.__init__(self, config, add_pooling_layer)

    def extract_sequence_outputs(self, outputs):
        return outputs['last_hidden_state']

    def extract_pooled_outputs(self, outputs):
        return outputs['pooler_output']

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                past_key_values=None,
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        return SbertModelTransform.forward(
            self, input_ids, attention_mask, token_type_ids, position_ids,
            head_mask, inputs_embeds, encoder_hidden_states,
            encoder_attention_mask, past_key_values, use_cache,
            output_attentions, output_hidden_states, return_dict, **kwargs)
