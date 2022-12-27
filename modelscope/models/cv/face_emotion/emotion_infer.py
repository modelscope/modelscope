# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from modelscope.utils.logger import get_logger
from .face_alignment.face_align import face_detection_PIL_v2

logger = get_logger()


def transform_PIL(img_pil):
    val_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return val_transforms(img_pil)


index2AU = [1, 2, 4, 6, 7, 10, 12, 15, 23, 24, 25, 26]
emotion_list = [
    'Neutral', 'Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise'
]


def inference(image, model, face_model, score_thre=0.5, GPU=0):
    image = image.cpu().numpy()
    image = Image.fromarray(image)
    face, bbox = face_detection_PIL_v2(image, face_model)
    if bbox is None:
        logger.warning('no face detected!')
        result = {'emotion_result': None, 'box': None}
        return result

    face = transform_PIL(face)
    face = face.unsqueeze(0)
    if torch.cuda.is_available():
        face = face.cuda(GPU)
    logits_AU, logits_emotion = model(face)
    logits_AU = torch.sigmoid(logits_AU)
    logits_emotion = nn.functional.softmax(logits_emotion, 1)

    _, index_list = logits_emotion.max(1)
    emotion_index = index_list[0].data.item()
    prob = logits_emotion[0][emotion_index]
    if prob > score_thre and emotion_index != 3:
        cur_emotion = emotion_list[emotion_index]
    else:
        cur_emotion = 'Neutral'

    logits_AU = logits_AU[0]
    au_ouput = torch.zeros_like(logits_AU)
    au_ouput[logits_AU >= score_thre] = 1
    au_ouput[logits_AU < score_thre] = 0

    au_ouput = au_ouput.int()

    cur_au_list = []
    for idx in range(au_ouput.shape[0]):
        if au_ouput[idx] == 1:
            au = index2AU[idx]
            cur_au_list.append(au)
    cur_au_list.sort()
    result = (cur_emotion, bbox)
    return result
