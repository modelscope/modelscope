import os
from typing import Any, Dict

from modelscope.utils.constant import Tasks
from ....utils.nlp.space.utils_dst import batch_to_device
from ...base import Model, Tensor
from ...builder import MODELS

__all__ = ['DialogStateTrackingModel']


@MODELS.register_module(Tasks.dialog_state_tracking, module_name=r'space')
class DialogStateTrackingModel(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
            model_cls (Optional[Any], optional): model loader, if None, use the
                default loader to load model weights, by default None.
        """

        super().__init__(model_dir, *args, **kwargs)

        from sofa.models.space import SpaceForDST, SpaceConfig
        self.model_dir = model_dir

        self.config = SpaceConfig.from_pretrained(self.model_dir)
        # self.model = SpaceForDST(self.config)
        self.model = SpaceForDST.from_pretrained(self.model_dir)
        self.model.to(self.config.device)

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
        import numpy as np
        import torch

        self.model.eval()
        batch = input['batch']
        batch = batch_to_device(batch, self.config.device)

        features = input['features']
        diag_state = input['diag_state']
        turn_itrs = [features[i.item()].guid.split('-')[2] for i in batch[9]]
        reset_diag_state = np.where(np.array(turn_itrs) == '0')[0]
        for slot in self.config.dst_slot_list:
            for i in reset_diag_state:
                diag_state[slot][i] = 0

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'input_mask': batch[1],
                'segment_ids': batch[2],
                'start_pos': batch[3],
                'end_pos': batch[4],
                'inform_slot_id': batch[5],
                'refer_id': batch[6],
                'diag_state': diag_state,
                'class_label_id': batch[8]
            }
            unique_ids = [features[i.item()].guid for i in batch[9]]
            values = [features[i.item()].values for i in batch[9]]
            input_ids_unmasked = [
                features[i.item()].input_ids_unmasked for i in batch[9]
            ]
            inform = [features[i.item()].inform for i in batch[9]]
            outputs = self.model(**inputs)

            # Update dialog state for next turn.
            for slot in self.config.dst_slot_list:
                updates = outputs[2][slot].max(1)[1]
                for i, u in enumerate(updates):
                    if u != 0:
                        diag_state[slot][i] = u

            print(outputs)

        return {
            'inputs': inputs,
            'outputs': outputs,
            'unique_ids': unique_ids,
            'input_ids_unmasked': input_ids_unmasked,
            'values': values,
            'inform': inform,
            'prefix': 'final'
        }
