from typing import Any, Dict, Optional

from modelscope.trainers.nlp.space.trainers.gen_trainer import MultiWOZTrainer
from modelscope.utils.constant import Tasks
from ...base import Model, Tensor
from ...builder import MODELS
from .model.generator import Generator
from .model.model_base import ModelBase

__all__ = ['DialogGenerationModel']


@MODELS.register_module(
    Tasks.dialog_generation, module_name=r'space-generation')
class DialogGenerationModel(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """

        super().__init__(model_dir, *args, **kwargs)
        self.model_dir = model_dir
        self.text_field = kwargs.pop('text_field')
        self.config = kwargs.pop('config')
        self.generator = Generator.create(self.config, reader=self.text_field)
        self.model = ModelBase.create(
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
            input (Dict[str, Any]): the preprocessed data

        Returns:
            Dict[str, np.ndarray]: results
                Example:
                    {
                        'predictions': array([1]), # lable 0-negative 1-positive
                        'probabilities': array([[0.11491239, 0.8850876 ]], dtype=float32),
                        'logits': array([[-0.53860897,  1.5029076 ]], dtype=float32) # true value
                    }
        """
        from numpy import array, float32
        import torch

        # turn_1 = {
        #     'user': [
        #         13, 1045, 2052, 2066, 1037, 10095, 2013, 3002, 2198, 1005,
        #         1055, 2267, 2000, 10733, 12570, 21713, 4487, 15474, 1012, 7
        #     ]
        # }
        # old_pv_turn_1 = {}

        turn_2 = {
            'user':
            [13, 1045, 2215, 2000, 2681, 2044, 2459, 1024, 2321, 1012, 7]
        }
        old_pv_turn_2 = {
            'labels': [[
                13, 1045, 2052, 2066, 1037, 10095, 2013, 3002, 2198, 1005,
                1055, 2267, 2000, 10733, 12570, 21713, 4487, 15474, 1012, 7
            ]],
            'resp': [
                14, 1045, 2052, 2022, 3407, 2000, 2393, 2007, 2115, 5227, 1010,
                2079, 2017, 2031, 1037, 2051, 2017, 2052, 2066, 2000, 2681,
                2030, 7180, 2011, 1029, 8
            ],
            'bspn': [
                15, 43, 7688, 10733, 12570, 21713, 4487, 15474, 6712, 3002,
                2198, 1005, 1055, 2267, 9
            ],
            'db': [19, 24, 21, 20],
            'aspn': [16, 43, 48, 2681, 7180, 10]
        }

        pv_turn = self.trainer.forward(turn=turn_2, old_pv_turn=old_pv_turn_2)

        return pv_turn
