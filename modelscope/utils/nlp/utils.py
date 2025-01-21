import os.path as osp

from modelscope.utils.hub import parse_label_mapping


def import_external_nltk_data(nltk_data_dir, package_name):
    """import external nltk_data, and extract nltk zip package.

    Args:
        nltk_data_dir (str): external nltk_data dir path, eg. /home/xx/nltk_data
        package_name (str): nltk package name, eg. tokenizers/punkt
    """
    import nltk
    nltk.data.path.append(nltk_data_dir)

    filepath = osp.join(nltk_data_dir, package_name + '.zip')
    zippath = osp.join(nltk_data_dir, package_name)
    packagepath = osp.dirname(zippath)
    if not osp.exists(zippath):
        import zipfile
        with zipfile.ZipFile(filepath) as zf:
            zf.extractall(osp.join(packagepath))


def parse_labels_in_order(model_dir=None, cfg=None, **kwargs):
    """Parse labels information in order.

    This is a helper function, used to get labels information in the correct order.
    1. The kw arguments listed in the method will in the first priority.
    2. Information in the cfg.dataset.train.labels will be used in the second priority (Compatible with old logic).
    3. Information in other files will be used then.

    Args:
        model_dir: The model_dir used to call `parse_label_mapping`.
        cfg: An optional cfg parsed and modified from the configuration.json.
        **kwargs: The user inputs into the method.

    Returns:
        The modified kwargs.
    """
    label2id = kwargs.pop('label2id', None)
    id2label = kwargs.pop('id2label', None)
    num_labels = kwargs.pop('num_labels', None)
    if label2id is None and id2label is not None:
        label2id = {label: id for id, label in id2label.items()}
    if label2id is None:
        if cfg is not None and cfg.safe_get(
                'dataset.train.labels') is not None:
            # An extra logic to parse labels from the dataset area.
            label2id = {
                label: idx
                for idx, label in enumerate(
                    cfg.safe_get('dataset.train.labels'))
            }
        elif model_dir is not None:
            label2id = parse_label_mapping(model_dir)

    if num_labels is None and label2id is not None:
        num_labels = len(label2id)
    if id2label is None and label2id is not None:
        id2label = {id: label for label, id in label2id.items()}
    if num_labels is not None:
        kwargs['num_labels'] = num_labels
    if label2id is not None:
        kwargs['label2id'] = label2id
    if id2label is not None:
        kwargs['id2label'] = id2label
    return kwargs
