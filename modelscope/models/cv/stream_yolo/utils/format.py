# Copyright (c) Alibaba, Inc. and its affiliates.
import math


def timestamp_format(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    time = '%02d:%02d:%06.3f' % (h, m, s)
    return time
