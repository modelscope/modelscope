# Part of the implementation is borrowed and modified from PackNet-SfM,
# made publicly available under the MIT License at https://github.com/TRI-ML/packnet-sfm
from modelscope.models.cv.video_depth_estimation.utils.types import (is_list,
                                                                     is_numpy,
                                                                     is_tensor)


def merge_outputs(*outputs):
    """
    Merges model outputs for logging

    Parameters
    ----------
    outputs : tuple of dict
        Outputs to be merged

    Returns
    -------
    output : dict
        Dictionary with a "metrics" key containing a dictionary with various metrics and
        all other keys that are not "loss" (it is handled differently).
    """
    ignore = ['loss']  # Keys to ignore
    combine = ['metrics']  # Keys to combine
    merge = {key: {} for key in combine}
    for output in outputs:
        # Iterate over all keys
        for key, val in output.items():
            # Combine these keys
            if key in combine:
                for sub_key, sub_val in output[key].items():
                    assert sub_key not in merge[key].keys(), \
                        'Combining duplicated key {} to {}'.format(sub_key, key)
                    merge[key][sub_key] = sub_val
            # Ignore these keys
            elif key not in ignore:
                assert key not in merge.keys(), \
                    'Adding duplicated key {}'.format(key)
                merge[key] = val
    return merge


def stack_batch(batch):
    """
    Stack multi-camera batches (B,N,C,H,W becomes BN,C,H,W)

    Parameters
    ----------
    batch : dict
        Batch

    Returns
    -------
    batch : dict
        Stacked batch
    """
    # If there is multi-camera information
    if len(batch['rgb'].shape) == 5:
        assert batch['rgb'].shape[
            0] == 1, 'Only batch size 1 is supported for multi-cameras'
        # Loop over all keys
        for key in batch.keys():
            # If list, stack every item
            if is_list(batch[key]):
                if is_tensor(batch[key][0]) or is_numpy(batch[key][0]):
                    batch[key] = [sample[0] for sample in batch[key]]
            # Else, stack single item
            else:
                batch[key] = batch[key][0]
    return batch
