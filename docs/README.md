## maintain docs
1. install requirements needed to build docs
    ```shell
    # in maas_lib root dir
    pip install requirements/docs.txt
    ```

2. build docs
    ```shell
    # in maas_lib/docs dir
    bash build_docs.sh
    ```

3. doc string format

    We adopt the google style docstring format as the standard, please refer to the following documents.
    1. Google Python style guide docstring [link](http://google.github.io/styleguide/pyguide.html#381-docstrings)
    2. Google docstring example [link](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
    3. sample：torch.nn.modules.conv [link](https://pytorch.org/docs/stable/_modules/torch/nn/modules/conv.html#Conv1d)
    4. load fucntion as an example：

    ```python
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
            >>> load('/path/of/your/file')  # file is storaged in disk
            >>> load('https://path/of/your/file')  # file is storaged in Internet
            >>> load('oss://path/of/your/file')  # file is storaged in petrel

        Returns:
            The content from the file.
        """
    ```
