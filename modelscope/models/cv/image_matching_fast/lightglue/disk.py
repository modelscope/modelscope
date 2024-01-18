import kornia
import torch

from .utils import Extractor


class DISK(Extractor):
    default_conf = {
        "weights": "depth",
        "max_num_keypoints": None,
        "desc_dim": 128,
        "nms_window_size": 5,
        "detection_threshold": 0.0,
        "pad_if_not_divisible": True,
    }

    preprocess_conf = {
        "resize": 1024,
        "grayscale": False,
    }

    required_data_keys = ["image"]

    def __init__(self, **conf) -> None:
        super().__init__(**conf)  # Update with default configuration.
        self.model = kornia.feature.DISK.from_pretrained(self.conf.weights)

    def forward(self, data: dict) -> dict:
        """Compute keypoints, scores, descriptors for image"""
        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        image = data["image"]
        if image.shape[1] == 1:
            image = kornia.color.grayscale_to_rgb(image)
        features = self.model(
            image,
            n=self.conf.max_num_keypoints,
            window_size=self.conf.nms_window_size,
            score_threshold=self.conf.detection_threshold,
            pad_if_not_divisible=self.conf.pad_if_not_divisible,
        )
        keypoints = [f.keypoints for f in features]
        scores = [f.detection_scores for f in features]
        descriptors = [f.descriptors for f in features]
        del features

        keypoints = torch.stack(keypoints, 0)
        scores = torch.stack(scores, 0)
        descriptors = torch.stack(descriptors, 0)

        return {
            "keypoints": keypoints.to(image).contiguous(),
            "keypoint_scores": scores.to(image).contiguous(),
            "descriptors": descriptors.to(image).contiguous(),
        }
