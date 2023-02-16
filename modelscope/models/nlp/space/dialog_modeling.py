# Copyright (c) Alibaba, Inc. and its affiliates.

import os
from typing import Dict

from modelscope.metainfo import Models
from modelscope.models import TorchModel
from modelscope.models.base import Tensor
from modelscope.models.builder import MODELS
from modelscope.models.nlp.space import SpaceGenerator, SpaceModelBase
from modelscope.preprocessors.nlp import MultiWOZBPETextField
from modelscope.utils.config import Config
from modelscope.utils.constant import ModelFile, Tasks

__all__ = ['SpaceForDialogModeling']


@MODELS.register_module(
    Tasks.task_oriented_conversation, module_name=Models.space_modeling)
class SpaceForDialogModeling(TorchModel):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (`str`):
                The model path.
            text_field (`BPETextField`, *optional*, defaults to `MultiWOZBPETextField`):
                The text field.
            config (`Config`, *optional*, defaults to config in model hub):
                The config.
        """

        super().__init__(model_dir, *args, **kwargs)
        from modelscope.trainers.nlp.space.trainer.gen_trainer import MultiWOZTrainer
        self.model_dir = model_dir
        self.config = kwargs.pop(
            'config',
            Config.from_file(
                os.path.join(self.model_dir, ModelFile.CONFIGURATION)))

        import torch
        self.config.use_gpu = True if (
            'device' not in kwargs or kwargs['device']
            == 'gpu') and torch.cuda.is_available() else False

        self.text_field = kwargs.pop(
            'text_field',
            MultiWOZBPETextField(config=self.config, model_dir=self.model_dir))
        self.generator = SpaceGenerator.create(
            self.config, reader=self.text_field)
        self.model = SpaceModelBase.create(
            model_dir=model_dir,
            config=self.config,
            reader=self.text_field,
            generator=self.generator)

        def to_tensor(array):
            """
            numpy array -> tensor
            """
            import torch
            array = torch.tensor(array)
            return array.cuda() if self.config.use_gpu else array

        self.trainer = MultiWOZTrainer(
            model=self.model,
            to_tensor=to_tensor,
            config=self.config,
            reader=self.text_field,
            evaluator=None)
        self.trainer.load()

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'labels': array([1,192,321,12]), # lable
                        'resp': array([293,1023,123,1123]), #vocab label for response
                        'bspn': array([123,321,2,24,1 ]),
                        'aspn': array([47,8345,32,29,1983]),
                        'db': array([19, 24, 20]),
                    }

        Examples:
            >>> from modelscope.hub.snapshot_download import snapshot_download
            >>> from modelscope.models.nlp import SpaceForDialogModeling
            >>> from modelscope.preprocessors import DialogModelingPreprocessor
            >>> cache_path = snapshot_download('damo/nlp_space_dialog-modeling')
            >>> preprocessor = DialogModelingPreprocessor(model_dir=cache_path)
            >>> model = SpaceForDialogModeling(model_dir=cache_path,
                    text_field=preprocessor.text_field,
                    config=preprocessor.config)
            >>> print(model(preprocessor({
                    'user_input': 'i would like a taxi from saint john \'s college to pizza hut fen ditton .',
                    'history': {}
                })))
        """

        first_turn = input['first_turn']
        batch = input['batch']
        prompt_id = input['prompt_id']
        labels = input['labels']
        old_pv_turn = input['history']

        pv_turn = self.trainer.forward(
            first_turn=first_turn,
            batch=batch,
            prompt_id=prompt_id,
            labels=labels,
            old_pv_turn=old_pv_turn)

        return pv_turn
