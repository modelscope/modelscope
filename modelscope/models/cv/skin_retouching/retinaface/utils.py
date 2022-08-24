# Implementation in this file is modifed from source code avaiable via https://github.com/ternaus/retinaface
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import torch


def load_checkpoint(file_path: Union[Path, str],
                    rename_in_layers: Optional[dict] = None) -> Dict[str, Any]:
    """Loads PyTorch checkpoint, optionally renaming layer names.
    Args:
        file_path: path to the torch checkpoint.
        rename_in_layers: {from_name: to_name}
            ex: {"model.0.": "",
                 "model.": ""}
    Returns:
    """
    checkpoint = torch.load(
        file_path, map_location=lambda storage, loc: storage)

    if rename_in_layers is not None:
        model_state_dict = checkpoint['state_dict']

        result = {}
        for key, value in model_state_dict.items():
            for key_r, value_r in rename_in_layers.items():
                key = re.sub(key_r, value_r, key)

            result[key] = value

        checkpoint['state_dict'] = result

    return checkpoint


def tensor_from_rgb_image(image: np.ndarray) -> torch.Tensor:
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image)


def vis_annotations(image: np.ndarray,
                    annotations: List[Dict[str, Any]]) -> np.ndarray:
    vis_image = image.copy()

    for annotation in annotations:
        landmarks = annotation['landmarks']

        colors = [(255, 0, 0), (128, 255, 0), (255, 178, 102), (102, 128, 255),
                  (0, 255, 255)]

        for landmark_id, (x, y) in enumerate(landmarks):
            vis_image = cv2.circle(
                vis_image, (x, y),
                radius=3,
                color=colors[landmark_id],
                thickness=3)

        x_min, y_min, x_max, y_max = annotation['bbox']

        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(
            vis_image, (x_min, y_min), (x_max, y_max),
            color=(0, 255, 0),
            thickness=2)
    return vis_image
