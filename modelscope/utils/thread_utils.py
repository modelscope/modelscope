# Copyright (c) Alibaba, Inc. and its affiliates.
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from functools import wraps
from modelscope.utils.logger import get_logger

logger = get_logger()


# TODO: to be fixed for nested progress bar.
def thread_executor(max_workers: int = min(8, os.cpu_count() + 4)):
    """
    A decorator to execute a function in a threaded manner using ThreadPoolExecutor.

    Args:
        max_workers (int): The maximum number of threads to use.

    Returns:
        function: A wrapped function that executes with threading and a progress bar.

    Examples:
        >>> from modelscope.utils.thread_utils import thread_executor
        >>> import time
        >>> @thread_executor(max_workers=8)
        ... def process_item(item, x, y):
        ...     # do something to single item
        ...     time.sleep(1)
        ...     return str(item) + str(x) + str(y)

        >>> items = [1, 2, 3]
        >>> process_item(items, x='abc', y='xyz')
    """

    def decorator(func):
        @wraps(func)
        def wrapper(iterable, *args, **kwargs):
            results = []
            # Create a tqdm progress bar with the total number of items to process
            with tqdm(total=len(iterable), desc=f'Processing {len(iterable)} items') as pbar:
                # Define a wrapper function to update the progress bar
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all tasks
                    futures = {executor.submit(func, item, *args, **kwargs): item for item in iterable}

                    # Update the progress bar as tasks complete
                    for future in as_completed(futures):
                        pbar.update(1)
                        results.append(future.result())
            return results

        return wrapper

    return decorator
