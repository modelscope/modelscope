# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import List

from modelscope.outputs import OutputKeys
from modelscope.pipelines.nlp import DialogStateTrackingPipeline


def tracking_and_print_dialog_states(
        test_case, pipelines: List[DialogStateTrackingPipeline]):
    import json
    pipelines_len = len(pipelines)
    history_states = [{}]
    utter = {}
    for step, item in enumerate(test_case):
        utter.update(item)
        result = pipelines[step % pipelines_len]({
            'utter':
            utter,
            'history_states':
            history_states
        })
        print(json.dumps(result))

        history_states.extend([result[OutputKeys.OUTPUT], {}])


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            batch_on_device.append(
                {k: v.to(device)
                 for k, v in element.items()})
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)
