import os.path as osp


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
