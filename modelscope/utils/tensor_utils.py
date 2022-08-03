# Copyright (c) Alibaba, Inc. and its affiliates.
# Part of the implementation is borrowed from huggingface/transformers.
from collections.abc import Mapping

import numpy as np


def torch_nested_numpify(tensors):
    import torch
    "Numpify `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_numpify(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        t = tensors.cpu()
        return t.numpy()
    return tensors


def torch_nested_detach(tensors):
    import torch
    "Detach `tensors` (even if it's a nested list/tuple of tensors)."
    if isinstance(tensors, (list, tuple)):
        return type(tensors)(torch_nested_detach(t) for t in tensors)
    if isinstance(tensors, torch.Tensor):
        return tensors.detach()
    return tensors


def torch_default_data_collator(features):
    # TODO @jiangnana.jnn refine this default data collator
    import torch
    first = features[0]

    if isinstance(first, Mapping):
        batch = {}
        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if 'label' in first and first['label'] is not None:
            label = first['label'].item() if isinstance(
                first['label'], torch.Tensor) else first['label']
            # the msdataset return a 0-dimension np.array with a single value, the following part handle this.
            if isinstance(label, np.ndarray):
                src_dtype = label[()].dtype
                dtype = torch.long if label[(
                )].dtype == np.int64 else torch.float
            else:
                src_dtype = type(label)
                dtype = torch.long if isinstance(label, int) else torch.float
            # add dtype to np.array to fix "TypeError: can't convert np.ndarray of type numpy.object_"
            batch['labels'] = torch.tensor(
                np.array([f['label'] for f in features], dtype=src_dtype),
                dtype=dtype)
        elif 'label_ids' in first and first['label_ids'] is not None:
            if isinstance(first['label_ids'], torch.Tensor):
                batch['labels'] = torch.stack(
                    [f['label_ids'] for f in features])
            else:
                dtype = torch.long if type(
                    first['label_ids'][0]) is int else torch.float
                batch['labels'] = torch.tensor(
                    [f['label_ids'] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ('label', 'label_ids'
                         ) and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack([d for f in features for d in f[k]])
                else:
                    batch[k] = torch.tensor(np.array([f[k] for f in features]))
    elif isinstance(first, tuple):
        batch = []
        for idx in range(len(first)):
            if isinstance(first[idx], torch.Tensor):
                batch.append(torch.stack([f[idx] for f in features]))
            else:
                batch.append(torch.tensor([f[idx] for f in features]))
    else:
        if isinstance(first, torch.Tensor):
            batch = torch.stack(features)
        else:
            batch = torch.tensor(features)

    return batch
