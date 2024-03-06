# The implementation is adopted from the CLIP4Clip implementation,
# made publicly available under Apache License, Version 2.0 at https://github.com/ArrowLuo/CLIP4Clip

import numpy as np


def get_retrieved_videos(sims, k):
    """
    Returns list of retrieved top k videos based on the sims matrix
    Args:
        sims: similar matrix.
        K: top k number of videos
    """
    argm = np.argsort(-sims, axis=1)
    topk = argm[:, :k].reshape(-1)
    retrieved_videos = np.unique(topk)
    return retrieved_videos


def get_index_to_normalize(sims, videos):
    """
    Returns list of indices to normalize from sims based on videos
    Args:
        sims: similar matrix.
        videos: video array.
    """
    argm = np.argsort(-sims, axis=1)[:, 0]
    result = np.array(list(map(lambda x: x in videos, argm)))
    result = np.nonzero(result)
    return result


def qb_norm(train_test, test_test, args):
    k = args.get('k', 1)
    beta = args.get('beta', 20)
    retrieved_videos = get_retrieved_videos(train_test, k)
    test_test_normalized = test_test
    train_test = np.exp(train_test * beta)
    test_test = np.exp(test_test * beta)

    normalizing_sum = np.sum(train_test, axis=0)
    index_for_normalizing = get_index_to_normalize(test_test, retrieved_videos)
    test_test_normalized[index_for_normalizing, :] = \
        np.divide(test_test[index_for_normalizing, :], normalizing_sum)
    return test_test_normalized
