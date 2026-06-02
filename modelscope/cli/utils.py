# Copyright (c) Alibaba, Inc. and its affiliates.
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def concurrent_download(download_fn, items, max_workers=8, item_name='item'):
    """Download multiple items concurrently with progress reporting.

    Args:
        download_fn: Callable that takes an item and returns
            (identifier, result_path, error_string_or_None).
        items: List of items to download.
        max_workers (int): Maximum concurrent workers.
        item_name (str): Display name for the item type.

    Returns:
        tuple: (succeeded_list, failed_list).
    """
    succeeded = []
    failed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_fn, item): item for item in items}
        for future in as_completed(futures):
            identifier, result_path, error = future.result()
            if error:
                failed.append((identifier, error))
                print(f'Failed to download {item_name} {identifier}: {error}')
            else:
                succeeded.append((identifier, result_path))
                print(f'Downloaded {item_name} {identifier} -> {result_path}')

    print(f'\nDownload complete: {len(succeeded)} succeeded, '
          f'{len(failed)} failed')
    if failed:
        print(f'Failed {item_name}s:')
        for identifier, error in failed:
            print(f'  {identifier}: {error}')
        sys.exit(1)

    return succeeded, failed
