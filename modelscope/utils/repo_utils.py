# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022-present, the HuggingFace Inc. team.

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Callable, Generator, Iterable, List, Optional, TypeVar, Union


T = TypeVar("T")
# Always ignore `.git` and `.cache/huggingface` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    ".git",
    ".git/*",
    "*/.git",
    "**/.git/**",
    ".cache/modelscope",
    ".cache/modelscope/*",
    "*/.cache/modelscope",
    "**/.cache/modelscope/**",
]
# Forbidden to commit these folders
FORBIDDEN_FOLDERS = [".git", ".cache"]


class RepoUtils:

    @staticmethod
    def filter_repo_objects(
        items: Iterable[T],
        *,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        key: Optional[Callable[[T], str]] = None,
    ) -> Generator[T, None, None]:
        """Filter repo objects based on an allowlist and a denylist.

        Input must be a list of paths (`str` or `Path`) or a list of arbitrary objects.
        In the later case, `key` must be provided and specifies a function of one argument
        that is used to extract a path from each element in iterable.

        Patterns are Unix shell-style wildcards which are NOT regular expressions. See
        https://docs.python.org/3/library/fnmatch.html for more details.

        Args:
            items (`Iterable`):
                List of items to filter.
            allow_patterns (`str` or `List[str]`, *optional*):
                Patterns constituting the allowlist. If provided, item paths must match at
                least one pattern from the allowlist.
            ignore_patterns (`str` or `List[str]`, *optional*):
                Patterns constituting the denylist. If provided, item paths must not match
                any patterns from the denylist.
            key (`Callable[[T], str]`, *optional*):
                Single-argument function to extract a path from each item. If not provided,
                the `items` must already be `str` or `Path`.

        Returns:
            Filtered list of objects, as a generator.

        Raises:
            :class:`ValueError`:
                If `key` is not provided and items are not `str` or `Path`.

        Example usage with paths:
        ```python
        >>> # Filter only PDFs that are not hidden.
        >>> list(RepoUtils.filter_repo_objects(
        ...     ["aaa.PDF", "bbb.jpg", ".ccc.pdf", ".ddd.png"],
        ...     allow_patterns=["*.pdf"],
        ...     ignore_patterns=[".*"],
        ... ))
        ["aaa.pdf"]
        ```
        """
        if isinstance(allow_patterns, str):
            allow_patterns = [allow_patterns]

        if isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]

        if allow_patterns is not None:
            allow_patterns = [RepoUtils._add_wildcard_to_directories(p) for p in allow_patterns]
        if ignore_patterns is not None:
            ignore_patterns = [RepoUtils._add_wildcard_to_directories(p) for p in ignore_patterns]

        if key is None:

            def _identity(item: T) -> str:
                if isinstance(item, str):
                    return item
                if isinstance(item, Path):
                    return str(item)
                raise ValueError(f"Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string.")

            key = _identity  # Items must be `str` or `Path`, otherwise raise ValueError

        for item in items:
            path = key(item)

            # Skip if there's an allowlist and path doesn't match any
            if allow_patterns is not None and not any(fnmatch(path, r) for r in allow_patterns):
                continue

            # Skip if there's a denylist and path matches any
            if ignore_patterns is not None and any(fnmatch(path, r) for r in ignore_patterns):
                continue

            yield item

    @staticmethod
    def _add_wildcard_to_directories(pattern: str) -> str:
        if pattern[-1] == "/":
            return pattern + "*"
        return pattern


@dataclass
class CommitInfo(str):
    """Data structure containing information about a newly created commit.

    Returned by any method that creates a commit on the Hub: [`create_commit`], [`upload_file`], [`upload_folder`],
    [`delete_file`], [`delete_folder`]. It inherits from `str` for backward compatibility but using methods specific
    to `str` is deprecated.

    Attributes:
        commit_url (`str`):
            Url where to find the commit.

        commit_message (`str`):
            The summary (first line) of the commit that has been created.

        commit_description (`str`):
            Description of the commit that has been created. Can be empty.

        oid (`str`):
            Commit hash id. Example: `"91c54ad1727ee830252e457677f467be0bfd8a57"`.

        pr_url (`str`, *optional*):
            Url to the PR that has been created, if any. Populated when `create_pr=True`
            is passed.

        pr_revision (`str`, *optional*):
            Revision of the PR that has been created, if any. Populated when
            `create_pr=True` is passed. Example: `"refs/pr/1"`.

        pr_num (`int`, *optional*):
            Number of the PR discussion that has been created, if any. Populated when
            `create_pr=True` is passed. Can be passed as `discussion_num` in
            [`get_discussion_details`]. Example: `1`.

        _url (`str`, *optional*):
            Legacy url for `str` compatibility. Can be the url to the uploaded file on the Hub (if returned by
            [`upload_file`]), to the uploaded folder on the Hub (if returned by [`upload_folder`]) or to the commit on
            the Hub (if returned by [`create_commit`]). Defaults to `commit_url`. It is deprecated to use this
            attribute. Please use `commit_url` instead.
    """

    commit_url: str
    commit_message: str
    commit_description: str
    oid: str
    pr_url: Optional[str] = None

    # Computed from `pr_url` in `__post_init__`
    pr_revision: Optional[str] = field(init=False)
    pr_num: Optional[str] = field(init=False)

    # legacy url for `str` compatibility (ex: url to uploaded file, url to uploaded folder, url to PR, etc.)
    _url: str = field(repr=False, default=None)  # type: ignore  # defaults to `commit_url`

    def __new__(cls, *args, commit_url: str, _url: Optional[str] = None, **kwargs):
        return str.__new__(cls, _url or commit_url)

