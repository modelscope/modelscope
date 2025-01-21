# Copyright (c) Alibaba, Inc. and its affiliates.

# Copyright (c) OpenMMLab. All rights reserved.
from io import BytesIO, StringIO
from pathlib import Path

from .file import File
from .format import JsonHandler, YamlHandler

format_handlers = {
    'json': JsonHandler(),
    'yaml': YamlHandler(),
    'yml': YamlHandler(),
}


def load(file, file_format=None, **kwargs):
    """Load data from json/yaml/pickle files.

    This method provides a unified api for loading data from serialized files.

    Args:
        file (str or :obj:`Path` or file-like object): Filename or a file-like
            object.
        file_format (str, optional): If not specified, the file format will be
            inferred from the file extension, otherwise use the specified one.
            Currently supported formats include "json", "yaml/yml".

    Examples:
        >>> load('/path/of/your/file')  # file is stored in disk
        >>> load('https://path/of/your/file')  # file is stored on internet
        >>> load('oss://path/of/your/file')  # file is stored in petrel

    Returns:
        The content from the file.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None and isinstance(file, str):
        file_format = file.split('.')[-1]
    if file_format not in format_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = format_handlers[file_format]
    if isinstance(file, str):
        if handler.text_mode:
            with StringIO(File.read_text(file)) as f:
                obj = handler.load(f, **kwargs)
        else:
            with BytesIO(File.read(file)) as f:
                obj = handler.load(f, **kwargs)
    elif hasattr(file, 'read'):
        obj = handler.load(file, **kwargs)
    else:
        raise TypeError('"file" must be a filepath str or a file-object')
    return obj


def dump(obj, file=None, file_format=None, **kwargs):
    """Dump data to json/yaml strings or files.

    This method provides a unified api for dumping data as strings or to files.

    Args:
        obj (any): The python object to be dumped.
        file (str or :obj:`Path` or file-like object, optional): If not
            specified, then the object is dumped to a str, otherwise to a file
            specified by the filename or file-like object.
        file_format (str, optional): Same as :func:`load`.

    Examples:
        >>> dump('hello world', '/path/of/your/file')  # disk
        >>> dump('hello world', 'oss://path/of/your/file')  # oss

    Returns:
        bool: True for success, False otherwise.
    """
    if isinstance(file, Path):
        file = str(file)
    if file_format is None:
        if isinstance(file, str):
            file_format = file.split('.')[-1]
        elif file is None:
            raise ValueError(
                'file_format must be specified since file is None')
    if file_format not in format_handlers:
        raise TypeError(f'Unsupported format: {file_format}')

    handler = format_handlers[file_format]
    if file is None:
        return handler.dump_to_str(obj, **kwargs)
    elif isinstance(file, str):
        if handler.text_mode:
            with StringIO() as f:
                handler.dump(obj, f, **kwargs)
                File.write_text(f.getvalue(), file)
        else:
            with BytesIO() as f:
                handler.dump(obj, f, **kwargs)
                File.write(f.getvalue(), file)
    elif hasattr(file, 'write'):
        handler.dump(obj, file, **kwargs)
    else:
        raise TypeError('"file" must be a filename str or a file-object')


def dumps(obj, format, **kwargs):
    """Dump data to json/yaml strings or files.

    This method provides a unified api for dumping data as strings or to files.

    Args:
        obj (any): The python object to be dumped.
        format (str, optional): Same as file_format :func:`load`.

    Examples:
        >>> dumps('hello world', 'json')  # json
        >>> dumps('hello world', 'yaml')  # yaml

    Returns:
        bool: True for success, False otherwise.
    """
    if format not in format_handlers:
        raise TypeError(f'Unsupported format: {format}')

    handler = format_handlers[format]
    return handler.dumps(obj, **kwargs)
