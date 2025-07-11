# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    print('TYPE_CHECKING...')
    from .batch_utils import (ctc_loss, ctc_prefix_beam_search, executor_cv,
                              executor_test, executor_train, is_sublist,
                              token_score_filter)
    from .det_utils import (compute_det, load_data_and_score, load_stats_file,
                            plot_det)
    from .file_utils import (make_pair, query_tokens_id, read_lexicon,
                             read_lists, read_token)
    from .model_utils import (average_model, convert_to_kaldi,
                              convert_to_pytorch, count_parameters,
                              load_checkpoint, save_checkpoint)
    from .runtime_utils import make_runtime_res

else:
    _import_structure = {
        'batch_utils': [
            'executor_train', 'executor_cv', 'executor_test',
            'token_score_filter', 'is_sublist', 'ctc_loss',
            'ctc_prefix_beam_search'
        ],
        'det_utils':
        ['load_data_and_score', 'load_stats_file', 'compute_det', 'plot_det'],
        'model_utils': [
            'count_parameters', 'load_checkpoint', 'save_checkpoint',
            'average_model', 'convert_to_kaldi', 'convert_to_pytorch'
        ],
        'file_utils': [
            'read_lists', 'make_pair', 'read_token', 'read_lexicon',
            'query_tokens_id'
        ],
        'runtime_utils': ['make_runtime_res'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
