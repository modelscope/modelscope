# Copyright (c) Alibaba, Inc. and its affiliates.

from urllib.parse import urlparse

import pandas as pd

from modelscope.utils.logger import get_logger

logger = get_logger()


def valid_url(url) -> bool:
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError as e:
        logger.warning(e)
        return False


def fetch_csv_with_url(csv_url: str) -> pd.DataFrame:
    """Fetch the csv content from url.

    Args:
        csv_url (str): The input url of csv data.

    Returns:
        A pandas DataFrame object which contains the csv content.
    """
    try:
        df = pd.read_csv(csv_url)
    except Exception as e:
        logger.error(f'Failed to fetch csv from url: {csv_url}')
        raise e

    return df
