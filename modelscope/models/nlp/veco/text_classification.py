# Copyright 2019 Facebook AI Research and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.
# Copyright 2021-2022 The Alibaba DAMO NLP Team Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from transformers import RobertaForSequenceClassification

from modelscope.metainfo import Models
from modelscope.models import Model, TorchModel
from modelscope.models.builder import MODELS
from modelscope.outputs import AttentionTextClassificationModelOutput
from modelscope.utils.constant import Tasks
from modelscope.utils.nlp.utils import parse_labels_in_order
from .configuration import VecoConfig


@MODELS.register_module(Tasks.nli, module_name=Models.veco)
@MODELS.register_module(
    Tasks.sentiment_classification, module_name=Models.veco)
@MODELS.register_module(Tasks.sentence_similarity, module_name=Models.veco)
@MODELS.register_module(Tasks.text_classification, module_name=Models.veco)
class VecoForSequenceClassification(TorchModel,
                                    RobertaForSequenceClassification):
    """Veco Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic
    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,
    pruning heads etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Preprocessor:
        This is the text classification model of Veco, the preprocessor of this model
        is `modelscope.preprocessors.TextClassificationTransformersPreprocessor`.

    Trainer:
        This model should be trained by dataset which has mixed languages,
        and evaluated by datasets of languages one by one.
        For example, if the training dataset is xnli (which has sub datasets of multiple languages), then you
        should mix the sub-datasets with the languages you want to train to one training dataset, and evaluate
        the model one sub-dataset by one sub-dataset of different languages.
        This procedure can be done by custom code. If you are using trainer of ModelScope,
        the `VecoTrainer` is suggested to use to train this model. This trainer overrides the basic evaluation
        loop, and will call the evaluation dataset one by one. Besides, this trainer will use the `VecoTaskDataset`
        to mix the input datasets to one, you can check the API Doc for the details.

        To check the complete example please
        view the unittest `test_veco_xnli` in `tests.trainers.test_finetune_sequence_classification.py`

    Parameters:
        config ([`VecoConfig`]): Model configuration class with all the parameters of the
            model. Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model
            weights.

    This class overrides [`RobertaForSequenceClassification`]. Please check the superclass for the
    appropriate documentation alongside usage examples.
    """

    config_class = VecoConfig

    def __init__(self, config, **kwargs):
        super().__init__(config.name_or_path, **kwargs)
        super(Model, self).__init__(config)

    def forward(self, *args, **kwargs):
        """
        Returns:
            Returns `modelscope.outputs.AttentionTextClassificationModelOutput`

        Examples:
            >>> from modelscope.models import Model
            >>> from modelscope.preprocessors import Preprocessor
            >>> model = Model.from_pretrained('damo/nlp_veco_fill-mask-large',
            >>>                               task='text-classification', num_labels=2)
            >>> preprocessor = Preprocessor.from_pretrained('damo/nlp_veco_fill-mask-large',
            >>>                                             label2id={'0': 0, '1': 1})
            >>> # Call the model, return some tensors
            >>> print(model(**preprocessor('这是个测试')))
            >>> # Call the pipeline, the result may be incorrect
            >>> from modelscope.pipelines import pipeline
            >>> pipeline_ins = pipeline('text-classification', pipeline_name='text-classification',
            >>>                         model=model, preprocessor=preprocessor)
            >>> print(pipeline_ins('这是个测试'))
        """

        kwargs['return_dict'] = True
        outputs = super(Model, self).forward(*args, **kwargs)
        return AttentionTextClassificationModelOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def _instantiate(cls, **kwargs):
        """Instantiate the model.

        Args:
            kwargs: Input args.
                    model_dir: The model dir used to load the checkpoint and the label information.
                    num_labels: An optional arg to tell the model how many classes to initialize.
                                    Method will call utils.parse_label_mapping if num_labels is not input.
                    label2id: An optional label2id mapping, which will cover the label2id in configuration (if exists).

        Returns:
            The loaded model, which is initialized by transformers.PreTrainedModel.from_pretrained
        """

        model_dir = kwargs.pop('model_dir', None)
        cfg = kwargs.pop('cfg', None)
        model_args = parse_labels_in_order(model_dir, cfg, **kwargs)

        if model_dir is None:
            config = VecoConfig(**model_args)
            model = cls(config)
        else:
            model = super(Model, cls).from_pretrained(
                pretrained_model_name_or_path=model_dir, **model_args)
        return model
