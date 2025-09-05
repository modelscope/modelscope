# Copyright (c) Alibaba, Inc. and its affiliates.
# Copyright 2022-present, the HuggingFace Inc. team.
# yapf: disable

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from modelscope.hub.utils.utils import convert_timestamp


def _parse_siblings(siblings_data: Optional[List[Dict[str, Any]]]) -> List['RepoSibling']:
    """
    Parse siblings data into RepoSibling objects.

    Args:
        siblings_data: Raw siblings data from API response, supporting both
                      uppercase (Path, Size, etc.) and lowercase (path, size, etc.) field names.

    Returns:
        List of RepoSibling objects.
    """
    if not siblings_data:
        return []

    return [
        RepoSibling(
            rfilename=sibling.get('Path') or sibling.get('path'),
            size=sibling.get('Size') or sibling.get('size'),
            blob_id=sibling.get('Sha256') or sibling.get('sha256'),
            type=sibling.get('Type') or sibling.get('type'),
            sha=sibling.get('Revision') or sibling.get('revision'),
            last_modified=convert_timestamp(sibling.get('CommittedDate') or sibling.get('committedDate')),
            lfs=BlobLfsInfo(
                size=sibling.get('Size') or sibling.get('size'),
                sha256=sibling.get('Sha256') or sibling.get('sha256'),
            )
        ) for sibling in siblings_data
    ]


@dataclass
class OrganizationInfo:
    """Organization information for a repository."""
    id: Optional[int]
    name: Optional[str]
    full_name: Optional[str]
    description: Optional[str]
    avatar: Optional[str]
    github_address: Optional[str]
    type: Optional[int]
    email: Optional[str]
    created_time: Optional[datetime.datetime]
    modified_time: Optional[datetime.datetime]

    def __init__(self, **kwargs):
        self.id = kwargs.pop('Id', None)
        self.name = kwargs.pop('Name', '')
        self.full_name = kwargs.pop('FullName', '')
        self.description = kwargs.pop('Description', '')
        self.avatar = kwargs.pop('Avatar', '')
        self.github_address = kwargs.pop('GithubAddress', '')
        self.type = kwargs.pop('Type', kwargs.pop('type', None))
        self.email = kwargs.pop('Email', kwargs.pop('email', ''))
        created_time = kwargs.pop('GmtCreated', kwargs.pop('created_time', None))
        self.created_time = convert_timestamp(created_time) if created_time else None
        modified_time = kwargs.pop('GmtModified', kwargs.pop('modified_time', None))
        self.modified_time = convert_timestamp(modified_time) if modified_time else None


@dataclass
class BlobLfsInfo:
    size: Optional[int] = None
    sha256: Optional[str] = None


@dataclass
class RepoSibling:
    """
    Contains basic information about a repo file inside a repo on the Hub.

    Attributes:
        rfilename (str): file name, relative to the repo root.
        size (`int`, *optional*): The file's size, in bytes.
        blob_id (`str`, *optional*): The file's git OID.
        lfs (`BlobLfsInfo`, *optional*): The file's LFS metadata.
        type (`str`, *optional*): The file's type.
        sha (`str`, *optional*): The file's latest commit SHA.
        last_modified (`datetime`, *optional*): The file's last modified time.
    """

    rfilename: str
    size: Optional[int] = None
    blob_id: Optional[str] = None
    type: Optional[str] = None
    sha: Optional[str] = None
    last_modified: Optional[datetime.datetime] = None
    lfs: Optional[BlobLfsInfo] = None


@dataclass
class ModelInfo:
    """
    Contains detailed information about a model on ModelScope Hub. This object is returned by [`model_info`].

    Attributes:
        id (`int`, *optional*): Model ID.
        name (`str`, *optional*): Model name.
        author (`str`, *optional*): Model author.
        chinese_name (`str`, *optional*): Chinese display name.
        visibility (`int`, *optional*): Visibility level (1=private, 5=public).
        is_published (`int`, *optional*): Whether the model is published.
        is_online (`int`, *optional*): Whether the model is online.
        already_star (`bool`, *optional*): Whether current user has starred this model.
        description (`str`, *optional*): Model description.
        license (`str`, *optional*): Model license.
        downloads (`int`, *optional*): Number of downloads.
        likes (`int`, *optional*): Number of likes.
        created_at (`datetime`, *optional*): Date of creation of the repo on the Hub.
        last_updated_time (`datetime`, *optional*): Last update timestamp.
        architectures (`List[str]`, *optional*): Model architectures.
        model_type (`List[str]`, *optional*): Model types.
        tasks (`List[Dict[str, Any]]`, *optional*): Supported tasks.
        readme_content (`str`, *optional*): README content.
        organization (`OrganizationInfo`, *optional*): Organization information.
        created_by (`str`, *optional*): Creator username.
        is_certification (`int`, *optional*): Certification status.
        approval_mode (`int`, *optional*): Approval mode.
        card_ready (`int`, *optional*): Whether model card is ready.
        backend_support (`str`, *optional*): Backend support information.
        model_infos (`Dict[str, Any]`, *optional*): Detailed model configuration information.
        tags (`List[str]`, *optional*): Model Tags.
        is_accessible (`int`, *optional*): Whether accessible.
        revision (`str`, *optional*): Revision/branch.
        related_arxiv_id (`List[str]`, *optional*): Related arXiv paper IDs.
        related_paper (`List[int]`, *optional*): Related papers.
        sha (`str`, *optional*): Latest commit SHA.
        last_modified (`datetime`, *optional*): Latest commit date.
        last_commit (`Dict[str, Any]`, *optional*): Latest commit information.
        siblings (List[RepoSibling], optional): Basic information about files that constitute the model.
    """

    id: Optional[int]
    name: Optional[str]
    author: Optional[str]
    chinese_name: Optional[str]
    visibility: Optional[int]
    is_published: Optional[int]
    is_online: Optional[int]
    already_star: Optional[bool]
    description: Optional[str]
    license: Optional[str]
    downloads: Optional[int]
    likes: Optional[int]
    created_at: Optional[datetime.datetime]
    last_updated_time: Optional[datetime.datetime]
    architectures: Optional[List[str]]
    model_type: Optional[List[str]]
    tasks: Optional[List[Dict[str, Any]]]
    readme_content: Optional[str]
    organization: Optional[OrganizationInfo]
    created_by: Optional[str]

    # Certification and approval
    is_certification: Optional[int]
    approval_mode: Optional[int]
    card_ready: Optional[int]

    # Model specific
    backend_support: Optional[str]
    model_infos: Optional[Dict[str, Any]]
    siblings: Optional[List[RepoSibling]]

    # Content and settings
    tags: Optional[List[str]]

    # Additional flags
    is_accessible: Optional[int]

    # Revision and version info
    revision: Optional[str]

    # External references
    related_arxiv_id: Optional[List[str]]
    related_paper: Optional[List[int]]

    # latest commit infomation
    last_commit: Optional[Dict[str, Any]]
    sha: Optional[str]
    last_modified: Optional[datetime.datetime]

    def __init__(self, **kwargs):
        self.id = kwargs.pop('Id', None)
        self.name = kwargs.pop('Name', '')
        self.chinese_name = kwargs.pop('ChineseName', '')
        self.visibility = kwargs.pop('Visibility', None)
        self.is_published = kwargs.pop('IsPublished', None)
        self.is_online = kwargs.pop('IsOnline', None)
        self.already_star = kwargs.pop('AlreadyStar', None)
        self.description = kwargs.pop('Description', '')
        self.license = kwargs.pop('License', '')
        self.downloads = kwargs.pop('Downloads', None)
        self.likes = kwargs.pop('Stars', None) or kwargs.pop('Likes', None)
        created_time = kwargs.pop('CreatedTime', None)
        self.created_at = convert_timestamp(created_time) if created_time else None
        last_updated_time = kwargs.pop('LastUpdatedTime', None)
        self.last_updated_time = convert_timestamp(last_updated_time) if last_updated_time else None
        self.architectures = kwargs.pop('Architectures', [])
        self.model_type = kwargs.pop('ModelType', [])
        self.tasks = kwargs.pop('Tasks', [])
        self.readme_content = kwargs.pop('ReadMeContent', '')
        org_data = kwargs.pop('Organization', None)
        self.organization = OrganizationInfo(**org_data) if org_data else None
        self.created_by = kwargs.pop('CreatedBy', None)
        self.is_certification = kwargs.pop('IsCertification', None)
        self.approval_mode = kwargs.pop('ApprovalMode', None)
        self.card_ready = kwargs.pop('CardReady', None)
        self.backend_support = kwargs.pop('BackendSupport', '{}')
        self.model_infos = kwargs.pop('ModelInfos', {})
        self.tags = kwargs.pop('Tags', [])
        self.is_accessible = kwargs.pop('IsAccessible', None)
        self.revision = kwargs.pop('Revision', '')
        self.related_arxiv_id = kwargs.pop('RelatedArxivId', [])
        self.related_paper = kwargs.pop('RelatedPaper', [])

        commits = kwargs.pop('commits', None) or kwargs.pop('Commits', None)
        if commits and hasattr(commits, 'commits') and commits.commits:
            last_commit = commits.commits[0]
            self.last_commit = last_commit.to_dict() if hasattr(last_commit, 'to_dict') else None
            self.sha = self.last_commit.get('id') if self.last_commit else None
            self.last_modified = convert_timestamp(self.last_commit.get('committed_date')) if self.last_commit else None
        else:
            self.last_commit = None
            self.sha = None
            self.last_modified = None
        self.author = kwargs.pop('author', '')

        siblings = kwargs.pop('siblings', None) or kwargs.pop('Siblings', None)
        self.siblings = _parse_siblings(siblings)

        # backward compatibility
        self.__dict__.update(kwargs)


@dataclass
class DatasetInfo:
    """
    Contains detailed information about a dataset on ModelScope Hub. This object is returned by [`dataset_info`].

    Attributes:
        id (`int`, *optional*)): Dataset ID.
        name (`str`, *optional*)): Dataset name.
        author (`str`, *optional*): Dataset owner (user or organization).
        chinese_name (`str`, *optional*): Chinese display name.
        visibility (`int`, *optional*)): Visibility level (1=private, 3=interal, 5=public).
        'internal' means visible to logged-in users only.
        already_star (`bool`, *optional*)): Whether current user has starred this dataset.
        description (`str`, *optional*): Dataset description.
        license (`str`, *optional*)): Dataset license.
        downloads (`int`, *optional*)): Number of downloads.
        likes (`int`, *optional*)): Number of likes.
        created_at (`int`, *optional*): Creation timestamp.
        last_updated_time (`int`, *optional*): Last update timestamp.
        readme_content (`str`, *optional*): README content.
        organization (`OrganizationInfo`, *optional*): Organization information.
        created_by (`str`, *optional*): Creator username.
        tags (`List[Dict[str, Any]]`): Dataset tags.
        last_commit (`Dict[str, Any]`, *optional*): Latest commit information.
        sha (`str`, *optional*): Latest commit SHA.
        last_modified (`datetime`, *optional*): Latest commit date.
        siblings (`List[RepoSibling]`, *optional*): Basic information about files in the dataset.
    """

    id: Optional[int]
    name: Optional[str]
    author: Optional[str]
    chinese_name: Optional[str]
    visibility: Optional[Literal[1, 3, 5]]
    already_star: Optional[bool]
    description: Optional[str]
    license: Optional[str]
    downloads: Optional[int]
    likes: Optional[int]
    created_at: Optional[datetime.datetime]
    last_updated_time: Optional[datetime.datetime]
    readme_content: Optional[str]
    organization: Optional[OrganizationInfo]
    created_by: Optional[str]
    tags: Optional[List[Dict[str, Any]]]
    last_commit: Optional[Dict[str, Any]]
    sha: Optional[str]
    last_modified: Optional[datetime.datetime]
    siblings: Optional[List[RepoSibling]]

    def __init__(self, **kwargs):
        self.id = kwargs.pop('Id', None)
        self.name = kwargs.pop('Name', '')
        self.author = kwargs.pop('author', kwargs.pop('Owner', None) or kwargs.pop('Namespace', None))
        self.chinese_name = kwargs.pop('ChineseName', '')
        self.visibility = kwargs.pop('Visibility', None)
        self.already_star = kwargs.pop('AlreadyStar', None)
        self.description = kwargs.pop('Description', '')
        self.likes = kwargs.pop('Likes', None) or kwargs.pop('Stars', None)
        self.license = kwargs.pop('License', '')
        self.downloads = kwargs.pop('Downloads', None)
        created_time = kwargs.pop('GmtCreate', None)
        self.created_at = convert_timestamp(created_time) if created_time else None
        last_updated_time = kwargs.pop('LastUpdatedTime', None)
        self.last_updated_time = convert_timestamp(last_updated_time) if last_updated_time else None
        self.readme_content = kwargs.pop('ReadMeContent', '')
        org_data = kwargs.pop('Organization', None)
        self.organization = OrganizationInfo(**org_data) if org_data else None
        self.created_by = kwargs.pop('CreatedBy', None)
        self.tags = kwargs.pop('Tags', [])

        commits = kwargs.pop('commits', None) or kwargs.pop('Commits', None)
        if commits and hasattr(commits, 'commits') and commits.commits:
            last_commit = commits.commits[0]
            self.last_commit = last_commit.to_dict() if hasattr(last_commit, 'to_dict') else None
            self.sha = self.last_commit.get('id') if self.last_commit else None
            self.last_modified = convert_timestamp(self.last_commit.get('committed_date')) if self.last_commit else None
        else:
            self.last_commit = None
            self.sha = None
            self.last_modified = None

        siblings = kwargs.pop('siblings', None) or kwargs.pop('Siblings', None)
        self.siblings = _parse_siblings(siblings)

        # backward compatibility
        self.__dict__.update(kwargs)
