# Copyright (c) Alibaba, Inc. and its affiliates.
import inspect
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import __version__ as transformers_version
    try:
        from transformers import Qwen2VLForConditionalGeneration
    except ImportError:
        pass

    try:
        from transformers import GPTQConfig
        from transformers import AwqConfig
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageToImage
    except ImportError:
        pass

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        pass

    try:
        from transformers import AutoModelForKeypointDetection
    except ImportError:
        pass

else:

    class UnsupportedAutoClass:

        def __init__(self, name: str):
            self.error_msg =\
                f'{name} is not supported with your installed Transformers version {transformers_version}. ' + \
                'Please update your Transformers by "pip install transformers -U".'

        def from_pretrained(self, pretrained_model_name_or_path, *model_args,
                            **kwargs):
            raise ImportError(self.error_msg)

        def from_config(self, cls, config):
            raise ImportError(self.error_msg)

    def user_agent(invoked_by=None):
        from modelscope.utils.constant import Invoke

        if invoked_by is None:
            invoked_by = Invoke.PRETRAINED
        uagent = '%s/%s' % (Invoke.KEY, invoked_by)
        return uagent

    def get_wrapped_class(module_class,
                          ignore_file_pattern=[],
                          file_filter=None,
                          **kwargs):
        """Get a custom wrapper class for  auto classes to download the models from the ModelScope hub
        Args:
            module_class: The actual module class
            ignore_file_pattern (`str` or `List`, *optional*, default to `None`):
                Any file pattern to be ignored in downloading, like exact file names or file extensions.
        Returns:
            The wrapper
        """
        default_ignore_file_pattern = ignore_file_pattern
        default_file_filter = file_filter

        class ClassWrapper(module_class):

            @classmethod
            def from_pretrained(cls, pretrained_model_name_or_path,
                                *model_args, **kwargs):

                from modelscope import snapshot_download
                from modelscope.utils.constant import DEFAULT_MODEL_REVISION, Invoke

                ignore_file_pattern = kwargs.pop('ignore_file_pattern',
                                                 default_ignore_file_pattern)
                subfolder = kwargs.pop('subfolder', default_file_filter)
                file_filter = None
                if subfolder:
                    file_filter = f'{subfolder}/*'
                if not os.path.exists(pretrained_model_name_or_path):
                    revision = kwargs.pop('revision', DEFAULT_MODEL_REVISION)
                    if file_filter is None:
                        model_dir = snapshot_download(
                            pretrained_model_name_or_path,
                            revision=revision,
                            ignore_file_pattern=ignore_file_pattern,
                            user_agent=user_agent())
                    else:
                        model_dir = os.path.join(
                            snapshot_download(
                                pretrained_model_name_or_path,
                                revision=revision,
                                ignore_file_pattern=ignore_file_pattern,
                                allow_file_pattern=file_filter,
                                user_agent=user_agent()), subfolder)
                else:
                    model_dir = pretrained_model_name_or_path

                module_obj = module_class.from_pretrained(
                    model_dir, *model_args, **kwargs)

                if module_class.__name__.startswith('AutoModel'):
                    module_obj.model_dir = model_dir
                return module_obj

        ClassWrapper.__name__ = module_class.__name__
        ClassWrapper.__qualname__ = module_class.__qualname__
        return ClassWrapper

    from .patcher import get_all_imported_modules
    all_imported_modules = get_all_imported_modules()
    all_available_modules = []
    large_file_free = ['config', 'tokenizer']
    for module in all_imported_modules:
        try:
            if (hasattr(module, 'from_pretrained')
                    and 'pretrained_model_name_or_path' in inspect.signature(
                        module.from_pretrained).parameters):
                if any(lf in module.__name__.lower()
                       for lf in large_file_free):
                    ignore_file_patterns = {
                        'ignore_file_pattern': [
                            r'\w+\.bin', r'\w+\.safetensors', r'\w+\.pth',
                            r'\w+\.pt', r'\w+\.h5'
                        ]
                    }
                else:
                    ignore_file_patterns = {}
                all_available_modules.append(
                    get_wrapped_class(module, **ignore_file_patterns))
        except (ImportError, AttributeError):
            pass

    for module in all_available_modules:
        globals()[module.__name__] = module
