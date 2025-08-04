# Copyright (c) Alibaba, Inc. and its affiliates.

import glob
import os
from enum import Enum
from typing import List, Optional

from modelscope.utils.logger import get_logger

# Get a logger instance from modelscope
logger = get_logger()

# Default AIGC model cover image
DEFAULT_AIGC_COVER_IMAGE = (
    'https://modelscope.cn/models/modelscope/modelscope_aigc_default_logo/resolve/master/'
    'aigc_default_logo.png')


class AigcModel:
    """
    Helper class to encapsulate AIGC-specific model creation parameters.
    """

    class AigcType(str, Enum):
        CHECKPOINT = 'Checkpoint'
        LORA = 'LoRA'
        VAE = 'VAE'

    class BaseModelType(str, Enum):
        SD_1_5 = 'SD_1_5'
        SD_XL = 'SD_XL'
        SD_3 = 'SD_3'
        FLUX_1 = 'FLUX_1'
        WAN_VIDEO_2_1_T2V_1_3_B = 'WAN_VIDEO_2_1_T2V_1_3_B'
        WAN_VIDEO_2_1_T2V_14_B = 'WAN_VIDEO_2_1_T2V_14_B'
        WAN_VIDEO_2_1_I2V_14_B = 'WAN_VIDEO_2_1_I2V_14_B'
        WAN_VIDEO_2_1_FLF2V_14_B = 'WAN_VIDEO_2_1_FLF2V_14_B'
        WAN_VIDEO_2_2_T2V_5_B = 'WAN_VIDEO_2_2_T2V_5_B'
        WAN_VIDEO_2_2_T2V_14_B = 'WAN_VIDEO_2_2_T2V_14_B'
        WAN_VIDEO_2_2_I2V_14_B = 'WAN_VIDEO_2_2_I2V_14_B'

    def __init__(self,
                 aigc_type: AigcType,
                 base_model_type: BaseModelType,
                 model_path: str,
                 tag: Optional[str] = 'v1.0',
                 tag_description: Optional[str] = 'this is an aigc model',
                 cover_images: Optional[List[str]] = None,
                 base_model_id: str = ''):
        """
        Initializes the AigcModel helper.

        Args:
            aigc_type (AigcType): AIGC model type.
                Valid values: Checkpoint, LoRA, VAE
            base_model_type (BaseModelType): Vision foundation model.
                Valid values: SD_1_5, SD_XL, SD_3, FLUX_1, WAN_VIDEO_2_1_T2V_1_3_B...
            model_path (str, required): The path of checkpoint/LoRA weights file (.safetensors) or folder
            tag (str, optional): Tag name for AIGC model, default 'v1.0'
            tag_description (str, optional): Tag description,
                default: 'this is a aigc model'
            cover_images (List[str], optional): List of cover image URLs,
                default: DEFAULT_AIGC_COVER_IMAGE
            base_model_id (str, optional): Base model name,
                default: '', e.g.'AI-ModelScope/FLUX.1-dev'
        """
        self.aigc_type = aigc_type
        self.base_model_type = base_model_type
        self.model_path = model_path
        self.tag = tag
        self.tag_description = tag_description
        self.cover_images = cover_images if cover_images is not None else [
            DEFAULT_AIGC_COVER_IMAGE
        ]
        self.base_model_id = base_model_id

        # Process model path and calculate weights information
        self._process_model_path()

    def _process_model_path(self):
        """Process model_path to extract weights information"""
        from modelscope.utils.file_utils import get_file_hash

        # Expand user path
        self.model_path = os.path.expanduser(self.model_path)

        if not os.path.exists(self.model_path):
            raise ValueError(f'Model path does not exist: {self.model_path}')

        target_file = None

        if os.path.isfile(self.model_path):
            target_file = self.model_path
            logger.info('Using file: %s', os.path.basename(target_file))
        elif os.path.isdir(self.model_path):
            safetensors_files = glob.glob(
                os.path.join(self.model_path, '*.safetensors'))

            if safetensors_files:
                target_file = safetensors_files[0]
                logger.info('✅ Found safetensors file: %s',
                            os.path.basename(target_file))
                if len(safetensors_files) > 1:
                    logger.warning(
                        'Multiple safetensors files found, using: %s',
                        os.path.basename(target_file))
                    logger.info(
                        'Other safetensors: %s',
                        [os.path.basename(f) for f in safetensors_files[1:]])
            else:
                # No .safetensors files, try to find any other file
                all_files = [
                    f for f in os.listdir(self.model_path)
                    if os.path.isfile(os.path.join(self.model_path, f))
                ]

                if all_files:
                    # Use the first available file
                    target_file = os.path.join(self.model_path, all_files[0])
                    logger.warning('No safetensors file found, using: %s',
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

            # Store weights information
            self.weights_filename = os.path.basename(target_file)
            self.weights_sha256 = hash_info['file_hash']
            self.weights_size = hash_info['file_size']
            self.target_file = target_file

    def upload_to_repo(self, api, model_id: str, token: Optional[str] = None):
        """Upload model files to repository."""
        logger.info('Uploading model to %s...', model_id)
        try:
            if os.path.isdir(self.model_path):
                # Upload entire folder
                logger.info('Uploading directory: %s', self.model_path)
                api.upload_folder(
                    repo_id=model_id,
                    folder_path=self.model_path,
                    token=token,
                    commit_message='Upload model folder for AIGC model')
            elif os.path.isfile(self.model_path):
                # Upload single file, target_file is guaranteed to be set by _process_model_path
                logger.info('Uploading file: %s', self.target_file)
                api.upload_file(
                    path_or_fileobj=self.target_file,
                    path_in_repo=self.weights_filename,
                    repo_id=model_id,
                    token=token,
                    commit_message=f'Upload {self.weights_filename} '
                    'for AIGC model')

            logger.info('✅ Successfully uploaded model to %s', model_id)
            return True
        except Exception as e:
            logger.warning('⚠️ Warning: Failed to upload model: %s', e)
            logger.warning(
                'You may need to upload the model manually after creation.')
            return False

    def to_dict(self) -> dict:
        """Converts the AIGC parameters to a dictionary suitable for API calls."""
        return {
            'aigc_type': self.aigc_type.value,
            'base_model_type': self.base_model_type.value,
            'tag': self.tag,
            'tag_description': self.tag_description,
            'cover_images': self.cover_images,
            'base_model_id': self.base_model_id,
            'model_path': self.model_path,
            'weights_filename': self.weights_filename,
            'weights_sha256': self.weights_sha256,
            'weights_size': self.weights_size
        }
