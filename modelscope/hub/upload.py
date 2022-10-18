# Copyright (c) Alibaba, Inc. and its affiliates.

import datetime
import os
import shutil
import tempfile
import uuid
from typing import Dict, Optional
from uuid import uuid4

from filelock import FileLock

from modelscope import __version__
from modelscope.hub.api import HubApi, ModelScopeConfig
from modelscope.hub.errors import InvalidParameter, NotLoginException
from modelscope.hub.git import GitCommandWrapper
from modelscope.hub.repository import Repository
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.logger import get_logger

logger = get_logger()


def upload_folder(model_id: str,
                  model_dir: str,
                  visibility: int = 0,
                  license: str = None,
                  chinese_name: Optional[str] = None,
                  commit_message: Optional[str] = None,
                  revision: Optional[str] = DEFAULT_MODEL_REVISION):
    """
    Upload model from a given directory to given repository. A valid model directory
    must contain a configuration.json file.

    This function upload the files in given directory to given repository. If the
    given repository is not exists in remote, it will automatically create it with
    given visibility, license and chinese_name parameters. If the revision is also
    not exists in remote repository, it will create a new branch for it.

    This function must be called before calling HubApi's login with a valid token
    which can be obtained from ModelScope's website.

    Args:
        model_id (`str`):
            The model id to be uploaded, caller must have write permission for it.
        model_dir(`str`):
            The Absolute Path of the finetune result.
        visibility(`int`, defaults to `0`):
            Visibility of the new created model(1-private, 5-public). If the model is
            not exists in ModelScope, this function will create a new model with this
            visibility and this parameter is required. You can ignore this parameter
            if you make sure the model's existence.
        license(`str`, defaults to `None`):
            License of the new created model(see License). If the model is not exists
            in ModelScope, this function will create a new model with this license
            and this parameter is required. You can ignore this parameter if you
            make sure the model's existence.
        chinese_name(`str`, *optional*, defaults to `None`):
            chinese name of the new created model.
        commit_message(`str`, *optional*, defaults to `None`):
            commit message of the push request.
        revision (`str`, *optional*, default to DEFAULT_MODEL_REVISION):
            which branch to push. If the branch is not exists, It will create a new
            branch and push to it.
    """
    if model_id is None:
        raise InvalidParameter('model_id cannot be empty!')
    if model_dir is None:
        raise InvalidParameter('model_dir cannot be empty!')
    if not os.path.exists(model_dir) or os.path.isfile(model_dir):
        raise InvalidParameter('model_dir must be a valid directory.')
    cfg_file = os.path.join(model_dir, ModelFile.CONFIGURATION)
    if not os.path.exists(cfg_file):
        raise ValueError(f'{model_dir} must contain a configuration.json.')
    cookies = ModelScopeConfig.get_cookies()
    if cookies is None:
        raise NotLoginException('Must login before upload!')
    files_to_save = os.listdir(model_dir)
    api = HubApi()
    try:
        api.get_model(model_id=model_id)
    except Exception:
        if visibility is None or license is None:
            raise InvalidParameter(
                'visibility and license cannot be empty if want to create new repo'
            )
        logger.info('Create new model %s' % model_id)
        api.create_model(
            model_id=model_id,
            visibility=visibility,
            license=license,
            chinese_name=chinese_name)
    tmp_dir = tempfile.mkdtemp()
    git_wrapper = GitCommandWrapper()
    try:
        repo = Repository(model_dir=tmp_dir, clone_from=model_id)
        branches = git_wrapper.get_remote_branches(tmp_dir)
        if revision not in branches:
            logger.info('Create new branch %s' % revision)
            git_wrapper.new_branch(tmp_dir, revision)
        git_wrapper.checkout(tmp_dir, revision)
        for f in files_to_save:
            if f[0] != '.':
                src = os.path.join(model_dir, f)
                if os.path.isdir(src):
                    shutil.copytree(src, os.path.join(tmp_dir, f))
                else:
                    shutil.copy(src, tmp_dir)
        if not commit_message:
            date = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            commit_message = '[automsg] push model %s to hub at %s' % (
                model_id, date)
        repo.push(commit_message=commit_message, branch=revision)
    except Exception:
        raise
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
