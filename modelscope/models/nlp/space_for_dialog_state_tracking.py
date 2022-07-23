import os
from typing import Any, Dict

from modelscope.utils.constant import Tasks
from ...metainfo import Models
from ...utils.nlp.space.utils_dst import batch_to_device
from ..base import Model, Tensor
from ..builder import MODELS

__all__ = ['SpaceForDialogStateTracking']


@MODELS.register_module(Tasks.dialog_state_tracking, module_name=Models.space)
class SpaceForDialogStateTracking(Model):

    def __init__(self, model_dir: str, *args, **kwargs):
        """initialize the test generation model from the `model_dir` path.

        Args:
            model_dir (str): the model path.
        """

        super().__init__(model_dir, *args, **kwargs)

        from sofa.models.space import SpaceForDST, SpaceConfig
        self.model_dir = model_dir

        self.config = SpaceConfig.from_pretrained(self.model_dir)
        self.model = SpaceForDST.from_pretrained(self.model_dir)

    def forward(self, input: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """return the result by the model

        Args:
            input (Dict[str, Tensor]): the preprocessed data

        Returns:
            Dict[str, Tensor]: results
                Example:
                    {
                        'inputs': dict(input_ids, input_masks,start_pos), # tracking states
                        'outputs': dict(slots_logits),
                        'unique_ids': str(test-example.json-0), # default value
                        'input_ids_unmasked': array([101, 7632, 1010,0,0,0])
                        'values': array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}]),
                        'inform':  array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}]),
                        'prefix': str('final'), #default value
                        'ds':  array([{'taxi-leaveAt': 'none', 'taxi-destination': 'none'}])
                    }
        """
        import numpy as np
        import torch

        self.model.eval()
        batch = input['batch']

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

        return {
            'inputs': inputs,
            'outputs': outputs,
            'unique_ids': unique_ids,
            'input_ids_unmasked': input_ids_unmasked,
            'values': values,
            'inform': inform,
            'prefix': 'final',
            'ds': input['ds']
        }
