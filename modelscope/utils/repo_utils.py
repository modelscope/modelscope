# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022-present, the HuggingFace Inc. team.
import base64
import functools
import hashlib
import io
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import (Any, BinaryIO, Callable, Generator, Iterable, Iterator,
                    List, Literal, Optional, TypeVar, Union)

from modelscope.hub.constants import DEFAULT_MODELSCOPE_DATA_ENDPOINT
from modelscope.hub.utils.utils import convert_timestamp
from modelscope.utils.file_utils import get_file_hash

T = TypeVar('T')
# Always ignore `.git` and `.cache/modelscope` folders in commits
DEFAULT_IGNORE_PATTERNS = [
    '.git',
    '.git/*',
    '*/.git',
    '**/.git/**',
    '.cache',
    '.cache/*',
    '*/.cache',
    '**/.cache/**',
]
# Forbidden to commit these folders
FORBIDDEN_FOLDERS = ['.git', '.cache']

UploadMode = Literal['lfs', 'normal']

DATASET_LFS_SUFFIX = [
    '.7z',
    '.aac',
    '.arrow',
    '.audio',
    '.bmp',
    '.bin',
    '.bz2',
    '.flac',
    '.ftz',
    '.gif',
    '.gz',
    '.h5',
    '.jack',
    '.jpeg',
    '.jpg',
    '.png',
    '.jsonl',
    '.joblib',
    '.lz4',
    '.msgpack',
    '.npy',
    '.npz',
    '.ot',
    '.parquet',
    '.pb',
    '.pickle',
    '.pcm',
    '.pkl',
    '.raw',
    '.rar',
    '.sam',
    '.tar',
    '.tgz',
    '.wasm',
    '.wav',
    '.webm',
    '.webp',
    '.zip',
    '.zst',
    '.tiff',
    '.mp3',
    '.mp4',
    '.ogg',
]

MODEL_LFS_SUFFIX = [
    '.7z',
    '.arrow',
    '.bin',
    '.bz2',
    '.ckpt',
    '.ftz',
    '.gz',
    '.h5',
    '.joblib',
    '.mlmodel',
    '.model',
    '.msgpack',
    '.npy',
    '.npz',
    '.onnx',
    '.ot',
    '.parquet',
    '.pb',
    '.pickle',
    '.pkl',
    '.pt',
    '.pth',
    '.rar',
    '.safetensors',
    '.tar',
    '.tflite',
    '.tgz',
    '.wasm',
    '.xz',
    '.zip',
    '.zst',
]


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

        allow_patterns = allow_patterns if allow_patterns else None
        ignore_patterns = ignore_patterns if ignore_patterns else None

        if isinstance(allow_patterns, str):
            allow_patterns = [allow_patterns]

        if isinstance(ignore_patterns, str):
            ignore_patterns = [ignore_patterns]

        if allow_patterns is not None:
            allow_patterns = [
                RepoUtils._add_wildcard_to_directories(p)
                for p in allow_patterns
            ]
        if ignore_patterns is not None:
            ignore_patterns = [
                RepoUtils._add_wildcard_to_directories(p)
                for p in ignore_patterns
            ]

        if key is None:

            def _identity(item: T) -> str:
                if isinstance(item, str):
                    return item
                if isinstance(item, Path):
                    return str(item)
                raise ValueError(
                    f'Please provide `key` argument in `filter_repo_objects`: `{item}` is not a string.'
                )

            key = _identity  # Items must be `str` or `Path`, otherwise raise ValueError

        for item in items:
            path = key(item)

            # Skip if there's an allowlist and path doesn't match any
            if allow_patterns is not None and not any(
                    fnmatch(path, r) for r in allow_patterns):
                continue

            # Skip if there's a denylist and path matches any
            if ignore_patterns is not None and any(
                    fnmatch(path, r) for r in ignore_patterns):
                continue

            yield item

    @staticmethod
    def _add_wildcard_to_directories(pattern: str) -> str:
        if pattern[-1] == '/':
            return pattern + '*'
        return pattern


@dataclass
class CommitInfo:
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

    """

    commit_url: str
    commit_message: str
    commit_description: str
    oid: str

    def to_dict(cls):
        return {
            'commit_url': cls.commit_url,
            'commit_message': cls.commit_message,
            'commit_description': cls.commit_description,
            'oid': cls.oid,
        }


@dataclass
class DetailedCommitInfo:
    """Detailed commit information from repository history API."""
    id: Optional[str]
    short_id: Optional[str]
    title: Optional[str]
    message: Optional[str]
    author_name: Optional[str]
    authored_date: Optional[datetime]
    author_email: Optional[str]
    committed_date: Optional[datetime]
    committer_name: Optional[str]
    committer_email: Optional[str]
    created_at: Optional[datetime]

    @classmethod
    def from_api_response(cls, data: dict) -> 'DetailedCommitInfo':
        """Create DetailedCommitInfo from API response data."""
        return cls(
            id=data.get('Id', ''),
            short_id=data.get('ShortId', ''),
            title=data.get('Title', ''),
            message=data.get('Message', ''),
            author_name=data.get('AuthorName', ''),
            authored_date=convert_timestamp(data.get('AuthoredDate', None)),
            author_email=data.get('AuthorEmail', ''),
            committed_date=convert_timestamp(data.get('CommittedDate', None)),
            committer_name=data.get('CommitterName', ''),
            committer_email=data.get('CommitterEmail', ''),
            created_at=convert_timestamp(data.get('CreatedAt', None)),
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'short_id': self.short_id,
            'title': self.title,
            'message': self.message,
            'author_name': self.author_name,
            'authored_date': self.authored_date,
            'author_email': self.author_email,
            'committed_date': self.committed_date,
            'committer_name': self.committer_name,
            'committer_email': self.committer_email,
            'created_at': self.created_at,
        }


@dataclass
class CommitHistoryResponse:
    """Response from commit history API."""
    commits: Optional[List[DetailedCommitInfo]]
    total_count: Optional[int]

    @classmethod
    def from_api_response(cls, data: dict) -> 'CommitHistoryResponse':
        """Create CommitHistoryResponse from API response data."""
        commits_data = data.get('Data', {}).get('Commit', [])

        if not commits_data:
            return cls(
                commits=[],
                total_count=0,
            )

        commits = [
            DetailedCommitInfo.from_api_response(commit)
            for commit in commits_data
        ]

        return cls(
            commits=commits,
            total_count=data.get('TotalCount', 0),
        )


@dataclass
class RepoUrl:

    url: Optional[str] = None
    namespace: Optional[str] = None
    repo_name: Optional[str] = None
    repo_id: Optional[str] = None
    repo_type: Optional[str] = None
    endpoint: Optional[str] = DEFAULT_MODELSCOPE_DATA_ENDPOINT

    def __repr__(self) -> str:
        return f"RepoUrl('{self}', endpoint='{self.endpoint}', repo_type='{self.repo_type}', repo_id='{self.repo_id}')"


def git_hash(data: bytes) -> str:
    """
    Computes the git-sha1 hash of the given bytes, using the same algorithm as git.

    This is equivalent to running `git hash-object`. See https://git-scm.com/docs/git-hash-object
    for more details.

    Note: this method is valid for regular files. For LFS files, the proper git hash is supposed to be computed on the
          pointer file content, not the actual file content. However, for simplicity, we directly compare the sha256 of
          the LFS file content when we want to compare LFS files.

    Args:
        data (`bytes`):
            The data to compute the git-hash for.

    Returns:
        `str`: the git-hash of `data` as an hexadecimal string.
    """
    _kwargs = {'usedforsecurity': False} if sys.version_info >= (3, 9) else {}
    sha1 = functools.partial(hashlib.sha1, **_kwargs)
    sha = sha1()
    sha.update(b'blob ')
    sha.update(str(len(data)).encode())
    sha.update(b'\0')
    sha.update(data)
    return sha.hexdigest()


@dataclass
class UploadInfo:
    """
    Dataclass holding required information to determine whether a blob
    should be uploaded to the hub using the LFS protocol or the regular protocol

    Args:
        sha256 (`str`):
            SHA256 hash of the blob
        size (`int`):
            Size in bytes of the blob
        sample (`bytes`):
            First 512 bytes of the blob
    """

    sha256: str
    size: int
    sample: bytes

    @classmethod
    def from_path(cls, path: str, file_hash_info: dict = None):
        file_hash_info = file_hash_info or get_file_hash(path)
        size = file_hash_info['file_size']
        sha = file_hash_info['file_hash']
        with open(path, 'rb') as f:
            sample = f.read(512)

        return cls(sha256=sha, size=size, sample=sample)

    @classmethod
    def from_bytes(cls, data: bytes, file_hash_info: dict = None):
        file_hash_info = file_hash_info or get_file_hash(data)
        sha = file_hash_info['file_hash']
        return cls(size=len(data), sample=data[:512], sha256=sha)

    @classmethod
    def from_fileobj(cls, fileobj: BinaryIO, file_hash_info: dict = None):
        file_hash_info: dict = file_hash_info or get_file_hash(fileobj)
        fileobj.seek(0, os.SEEK_SET)
        sample = fileobj.read(512)
        fileobj.seek(0, os.SEEK_SET)
        return cls(
            sha256=file_hash_info['file_hash'],
            size=file_hash_info['file_size'],
            sample=sample)


@dataclass
class CommitOperationAdd:
    """Data structure containing information about a file to be added to a commit."""

    path_in_repo: str
    path_or_fileobj: Union[str, Path, bytes, BinaryIO]
    upload_info: UploadInfo = field(init=False, repr=False)
    file_hash_info: dict = field(default_factory=dict)

    # Internal attributes

    # set to "lfs" or "regular" once known
    _upload_mode: Optional[UploadMode] = field(
        init=False, repr=False, default=None)

    # set to True if .gitignore rules prevent the file from being uploaded as LFS
    # (server-side check)
    _should_ignore: Optional[bool] = field(
        init=False, repr=False, default=None)

    # set to the remote OID of the file if it has already been uploaded
    # useful to determine if a commit will be empty or not
    _remote_oid: Optional[str] = field(init=False, repr=False, default=None)

    # set to True once the file has been uploaded as LFS
    _is_uploaded: bool = field(init=False, repr=False, default=False)

    # set to True once the file has been committed
    _is_committed: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        """Validates `path_or_fileobj` and compute `upload_info`."""

        self.path_in_repo = _validate_path_in_repo(self.path_in_repo)

        # Validate `path_or_fileobj` value
        if isinstance(self.path_or_fileobj, Path):
            self.path_or_fileobj = str(self.path_or_fileobj)
        if isinstance(self.path_or_fileobj, str):
            path_or_fileobj = os.path.normpath(
                os.path.expanduser(self.path_or_fileobj))
            if not os.path.isfile(path_or_fileobj):
                raise ValueError(
                    f"Provided path: '{path_or_fileobj}' is not a file on the local file system"
                )
        elif not isinstance(self.path_or_fileobj, (io.BufferedIOBase, bytes)):
            raise ValueError(
                'path_or_fileobj must be either an instance of str, bytes or'
                ' io.BufferedIOBase. If you passed a file-like object, make sure it is'
                ' in binary mode.')
        if isinstance(self.path_or_fileobj, io.BufferedIOBase):
            try:
                self.path_or_fileobj.tell()
                self.path_or_fileobj.seek(0, os.SEEK_CUR)
            except (OSError, AttributeError) as exc:
                raise ValueError(
                    'path_or_fileobj is a file-like object but does not implement seek() and tell()'
                ) from exc

        # Compute "upload_info" attribute
        if isinstance(self.path_or_fileobj, str):
            self.upload_info = UploadInfo.from_path(self.path_or_fileobj,
                                                    self.file_hash_info)
        elif isinstance(self.path_or_fileobj, bytes):
            self.upload_info = UploadInfo.from_bytes(self.path_or_fileobj,
                                                     self.file_hash_info)
        else:
            self.upload_info = UploadInfo.from_fileobj(self.path_or_fileobj,
                                                       self.file_hash_info)

    @contextmanager
    def as_file(self) -> Iterator[BinaryIO]:
        """
        A context manager that yields a file-like object allowing to read the underlying
        data behind `path_or_fileobj`.
        """
        if isinstance(self.path_or_fileobj, str) or isinstance(
                self.path_or_fileobj, Path):
            with open(self.path_or_fileobj, 'rb') as file:
                yield file
        elif isinstance(self.path_or_fileobj, bytes):
            yield io.BytesIO(self.path_or_fileobj)
        elif isinstance(self.path_or_fileobj, io.BufferedIOBase):
            prev_pos = self.path_or_fileobj.tell()
            yield self.path_or_fileobj
            self.path_or_fileobj.seek(prev_pos, 0)

    def b64content(self) -> bytes:
        """
        The base64-encoded content of `path_or_fileobj`

        Returns: `bytes`
        """
        with self.as_file() as file:
            return base64.b64encode(file.read())

    @property
    def _local_oid(self) -> Optional[str]:
        """Return the OID of the local file.

        This OID is then compared to `self._remote_oid` to check if the file has changed compared to the remote one.
        If the file did not change, we won't upload it again to prevent empty commits.

        For LFS files, the OID corresponds to the SHA256 of the file content (used a LFS ref).
        For regular files, the OID corresponds to the SHA1 of the file content.
        Note: this is slightly different to git OID computation since the oid of an LFS file is usually the git-SHA1
            of the pointer file content (not the actual file content). However, using the SHA256 is enough to detect
            changes and more convenient client-side.
        """
        if self._upload_mode is None:
            return None
        elif self._upload_mode == 'lfs':
            return self.upload_info.sha256
        else:
            # Regular file => compute sha1
            # => no need to read by chunk since the file is guaranteed to be <=5MB.
            with self.as_file() as file:
                return git_hash(file.read())


def _validate_path_in_repo(path_in_repo: str) -> str:
    # Validate `path_in_repo` value to prevent a server-side issue
    if path_in_repo.startswith('/'):
        path_in_repo = path_in_repo[1:]
    if path_in_repo == '.' or path_in_repo == '..' or path_in_repo.startswith(
            '../'):
        raise ValueError(
            f"Invalid `path_in_repo` in CommitOperation: '{path_in_repo}'")
    if path_in_repo.startswith('./'):
        path_in_repo = path_in_repo[2:]
    for forbidden in FORBIDDEN_FOLDERS:
        if any(part == forbidden for part in path_in_repo.split('/')):
            raise ValueError(
                f"Invalid `path_in_repo` in CommitOperation: cannot update files under a '{forbidden}/' folder (path:"
                f" '{path_in_repo}').")
    return path_in_repo


CommitOperation = Union[CommitOperationAdd, ]
