# Copyright (c) Alibaba, Inc. and its affiliates.

from modelscope.utils.import_utils import is_tf_available

if is_tf_available():
    from .cartoon_translation_exporter import CartoonTranslationExporter
