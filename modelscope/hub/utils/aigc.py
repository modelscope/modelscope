# Copyright (c) Alibaba, Inc. and its affiliates.
import glob
import os
from typing import List, Optional

import requests
from tqdm.auto import tqdm

from modelscope.hub.utils.utils import MODELSCOPE_URL_SCHEME, get_domain
from modelscope.utils.logger import get_logger

logger = get_logger()

# Default AIGC model cover image
DEFAULT_AIGC_COVER_IMAGE = (
    'https://modelscope.cn/models/modelscope/modelscope_aigc_default_logo/resolve/master/'
    'aigc_default_logo.png')


class AigcModel:
    """
    Helper class to encapsulate AIGC-specific model creation parameters.

    This class can be initialized directly with parameters, or loaded from a
    JSON configuration file using the `from_json_file` classmethod.

    Example of direct initialization:
        >>> aigc_model = AigcModel(
        ...     aigc_type='Checkpoint',
        ...     base_model_type='SD_XL',
        ...     model_path='/path/to/your/model.safetensors'
        ...     base_model_id='AI-ModelScope/FLUX.1-dev'
        ... )

    Example of loading from a JSON file:
        `config.json`:
        {
            "model_path": "/path/to/your/model.safetensors",
            "aigc_type": "Checkpoint",
            "base_model_type": "SD_XL",
            "base_model_id": "AI-ModelScope/FLUX.1-dev"
        }

        >>> aigc_model = AigcModel.from_json_file('config.json')
    """

    AIGC_TYPES = {'Checkpoint', 'LoRA', 'VAE'}

    # Supported base model types for reference
    BASE_MODEL_TYPES = {
        'SD_1_5', 'SD_XL', 'SD_3', 'FLUX_1', 'WAN_VIDEO_2_1_T2V_1_3_B',
        'WAN_VIDEO_2_1_T2V_14_B', 'WAN_VIDEO_2_1_I2V_14_B',
        'WAN_VIDEO_2_1_FLF2V_14_B', 'WAN_VIDEO_2_2_T2V_5_B',
        'WAN_VIDEO_2_2_T2V_14_B', 'WAN_VIDEO_2_2_I2V_14_B', 'QWEN_IMAGE_20B'
    }

    def __init__(self,
                 aigc_type: str,
                 base_model_type: str,
                 model_path: str,
                 base_model_id: str = '',
                 revision: Optional[str] = 'v1.0',
                 description: Optional[str] = 'this is an aigc model',
                 cover_images: Optional[List[str]] = None,
                 path_in_repo: Optional[str] = ''):
        """
        Initializes the AigcModel helper.

        Args:
            model_path (str): The path of checkpoint/LoRA weight file or folder.
            aigc_type (str): AIGC model type. Recommended: 'Checkpoint', 'LoRA', 'VAE'.
            base_model_type (str): Vision foundation model type. Recommended values are in BASE_MODEL_TYPES.
            revision (str, optional): Revision for the AIGC model. Defaults to 'v1.0'.
            description (str, optional): Model description. Defaults to 'this is an aigc model'.
            cover_images (List[str], optional): List of cover image URLs.
            base_model_id (str, optional): Base model name. e.g., 'AI-ModelScope/FLUX.1-dev'.
            path_in_repo (str, optional): Path in the repository.
                Note: Auto-upload during AIGC create is temporarily disabled by server. This parameter
                will not take effect at creation time.
        """
        self.model_path = model_path
        self.aigc_type = aigc_type
        self.base_model_type = base_model_type
        self.revision = revision
        self.description = description
        self.cover_images = cover_images if cover_images is not None else [
            DEFAULT_AIGC_COVER_IMAGE
        ]
        self.base_model_id = base_model_id
        self.path_in_repo = path_in_repo

        # Validate types and provide warnings
        self._validate_aigc_type()
        self._validate_base_model_type()

        # Process model path and calculate weights information
        self._process_model_path()

    def _validate_aigc_type(self):
        """Validate aigc_type and provide a warning for unsupported types."""
        if self.aigc_type not in self.AIGC_TYPES:
            supported_types = ', '.join(sorted(self.AIGC_TYPES))
            logger.warning(f'Unsupported aigc_type: "{self.aigc_type}". '
                           f'Recommended values: {supported_types}. '
                           'Custom values are allowed but may cause issues.')

    def _validate_base_model_type(self):
        """Validate base_model_type and provide warning for unsupported types."""
        if self.base_model_type not in self.BASE_MODEL_TYPES:
            supported_types = ', '.join(sorted(self.BASE_MODEL_TYPES))
            logger.warning(
                f'Your base_model_type: "{self.base_model_type}" may not be supported. '
                f'Recommended values: {supported_types}. '
                f'Custom values are allowed but may cause issues. ')

    def _process_model_path(self):
        """Process model_path to extract weight information"""
        from modelscope.utils.file_utils import get_file_hash

        # Expand user path
        self.model_path = os.path.expanduser(self.model_path)

        if not os.path.exists(self.model_path):
            raise ValueError(f'Model path does not exist: {self.model_path}')

        if os.path.isfile(self.model_path):
            target_file = self.model_path
            logger.info('Using file: %s', os.path.basename(target_file))
        elif os.path.isdir(self.model_path):
            # Validate top-level directory: it must not be empty; and if it has files,
            # they must not be only the common placeholder files
            top_entries = os.listdir(self.model_path)
            if len(top_entries) == 0:
                raise ValueError(
                    f'Directory is empty: {self.model_path}. '
                    f'Please place at least one model file at the top level (e.g., .safetensors/.pth/.bin).'
                )

            top_files = [
                name for name in top_entries
                if os.path.isfile(os.path.join(self.model_path, name))
            ]
            placeholder_names = {
                '.gitattributes', 'configuration.json', 'readme.md'
            }
            if top_files:
                normalized = {name.lower() for name in top_files}
                if normalized.issubset(placeholder_names):
                    raise ValueError(
                        'Top-level directory contains only [.gitattributes, configuration.json, README.md]. '
                        'Please place additional model files at the top level (e.g., .safetensors/.pth/.bin).'
                    )

            # Priority order for metadata file: safetensors -> pth -> bin -> first file
            file_extensions = ['.safetensors', '.pth', '.bin']
            target_file = None

            for ext in file_extensions:
                files = glob.glob(os.path.join(self.model_path, f'*{ext}'))
                if files:
                    target_file = files[0]
                    logger.info(f'Found {ext} file: %s',
                                os.path.basename(target_file))
                    if len(files) > 1:
                        logger.warning(
                            f'Multiple {ext} files found, using: %s for metadata',
                            os.path.basename(target_file))
                        logger.info(f'Other {ext} files: %s',
                                    [os.path.basename(f) for f in files[1:]])
                    break

            # If no preferred files found, use the first available file
            if not target_file:
                all_files = [
                    f for f in os.listdir(self.model_path)
                    if os.path.isfile(os.path.join(self.model_path, f))
                ]

                if all_files:
                    target_file = os.path.join(self.model_path, all_files[0])
                    logger.warning(
                        'No safetensors/pth/bin files found, using: %s for metadata',
                        os.path.basename(target_file))
                    logger.info('Available files: %s', all_files)
                else:
                    raise ValueError(
                        f'No files found in directory: {self.model_path}. '
                        f'AIGC models require at least one model file (.safetensors recommended).'
                    )

        else:
            raise ValueError(
                f'Model path must be a file or directory: {self.model_path}')

        if target_file:
            # Calculate file hash and size for the target file
            logger.info('Computing hash and size for %s...', target_file)
            hash_info = get_file_hash(target_file)

            # Store weight information
            self.weight_filename = os.path.basename(target_file)
            self.weight_sha256 = hash_info['file_hash']
            self.weight_size = hash_info['file_size']
            self.target_file = target_file

    def upload_to_repo(self, api, model_id: str, token: Optional[str] = None):
        """Upload model files to repository."""
        logger.info('Uploading model to %s...', model_id)
        try:
            if os.path.isdir(self.model_path):
                # Upload entire folder with path_in_repo support
                logger.info('Uploading directory: %s', self.model_path)
                api.upload_folder(
                    revision=self.revision,
                    repo_id=model_id,
                    folder_path=self.model_path,
                    path_in_repo=self.path_in_repo,
                    token=token,
                    commit_message='Upload model folder for AIGC model')
            elif os.path.isfile(self.model_path):
                # Upload single file, target_file is guaranteed to be set by _process_model_path
                logger.info('Uploading file: %s', self.target_file)
                api.upload_file(
                    revision=self.revision,
                    path_or_fileobj=self.target_file,
                    path_in_repo=self.path_in_repo + '/' + self.weight_filename
                    if self.path_in_repo else self.weight_filename,
                    repo_id=model_id,
                    token=token,
                    commit_message=f'Upload {self.weight_filename} '
                    'for AIGC model')

            logger.info('Successfully uploaded model to %s', model_id)
            return True
        except Exception as e:
            logger.warning('Warning: Failed to upload model: %s', e)
            logger.warning(
                'You may need to upload the model manually after creation.')
            return False

    def preupload_weights(self,
                          *,
                          cookies: Optional[object] = None,
                          timeout: int = 300,
                          headers: Optional[dict] = None) -> None:
        """Pre-upload aigc model weights to the LFS server.

        Server may require the sha256 of weights to be registered before creation.
        This method streams the weight file so the sha gets registered.

        Args:
            cookies: Optional requests-style cookies (CookieJar/dict). If provided, preferred.
            timeout: Request timeout seconds.
            headers: Optional headers.
        """
        domain: str = get_domain()
        base_url: str = f'{MODELSCOPE_URL_SCHEME}lfs.{domain.lstrip("www.")}'
        url: str = f'{base_url}/api/v1/models/aigc/weights'

        file_path = getattr(self, 'target_file', None) or self.model_path
        file_path = os.path.abspath(os.path.expanduser(file_path))
        if not os.path.isfile(file_path):
            raise ValueError(f'Pre-upload expects a file, got: {file_path}')

        cookies = dict(cookies) if cookies else None
        if cookies is None:
            raise ValueError('Token does not exist, please login first.')

        headers.update({'Cookie': f"m_session_id={cookies['m_session_id']}"})

        file_size = os.path.getsize(file_path)

        def read_in_chunks(file_object,
                           pbar,
                           chunk_size: int = 1 * 1024 * 1024):
            while True:
                ck = file_object.read(chunk_size)
                if not ck:
                    break
                pbar.update(len(ck))
                yield ck

        with tqdm(
                total=file_size,
                unit='B',
                unit_scale=True,
                dynamic_ncols=True,
                desc='[Pre-uploading] ') as pbar:
            with open(file_path, 'rb') as f:
                r = requests.put(
                    url,
                    headers=headers,
                    data=read_in_chunks(f, pbar),
                    timeout=timeout,
                )
        try:
            resp = r.json()
        except requests.exceptions.JSONDecodeError:
            r.raise_for_status()
            return
        # If JSON body returned, try best-effort check
        if isinstance(resp, dict) and resp.get('Success') is False:
            msg = resp.get('Message', 'unknown error')
            raise RuntimeError(f'Pre-upload failed: {msg}')

    def to_dict(self) -> dict:
        """Converts the AIGC parameters to a dictionary suitable for API calls."""
        return {
            'aigc_type': self.aigc_type,
            'base_model_type': self.base_model_type,
            'revision': self.revision,
            'description': self.description,
            'cover_images': self.cover_images,
            'base_model_id': self.base_model_id,
            'model_path': self.model_path,
            'weight_filename': self.weight_filename,
            'weight_sha256': self.weight_sha256,
            'weight_size': self.weight_size
        }

    @classmethod
    def from_json_file(cls, json_path: str):
        """
        Creates an AigcModel instance from a JSON configuration file.

        Args:
            json_path (str): The path to the JSON configuration file.

        Returns:
            AigcModel: An instance of the AigcModel.
        """
        import json
        json_path = os.path.expanduser(json_path)
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f'JSON config file not found at: {json_path}')

        with open(json_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Ensure required fields are present
        required_fields = [
            'model_path', 'aigc_type', 'base_model_type', 'base_model_id'
        ]
        for field in required_fields:
            if field not in config:
                raise ValueError(
                    f"Missing required field in JSON config: '{field}'")

        return cls(**config)
