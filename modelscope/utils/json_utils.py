# Copyright (c) Alibaba, Inc. and its affiliates.

import json
import numpy as np


class EnhancedEncoder(json.JSONEncoder):
    """ Enhanced json encoder for not supported types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
