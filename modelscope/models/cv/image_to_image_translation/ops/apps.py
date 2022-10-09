# Copyright 2021-2022 The Alibaba Fundamental Vision Team Authors. All rights reserved.
# APPs that facilitate the use of pretrained neural networks.

import os.path as osp

import artist.data as data
import artist.models as models
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn.functional as F
import torchvision.transforms as T
from artist import DOWNLOAD_TO_CACHE
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .utils import parallel, read_image

__all__ = [
    'FeatureExtractor', 'Classifier', 'Text2Image', 'Sole2Shoe', 'ImageParser',
    'TextImageMatch', 'taobao_feature_extractor', 'singleton_classifier',
    'orientation_classifier', 'fashion_text2image', 'mindalle_text2image',
    'sole2shoe', 'sole_parser', 'sod_foreground_parser',
    'fashion_text_image_match'
]


class ImageFolder(Dataset):

    def __init__(self, paths, transforms=None):
        self.paths = paths
        self.transforms = transforms

    def __getitem__(self, index):
        img = read_image(self.paths[index])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self.paths)


class FeatureExtractor(object):

    def __init__(
        self,
        model='InceptionV1',
        checkpoint='models/inception-v1/1218shoes.v9_7.140.0.1520000',
        resolution=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=64,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device

        # init model
        self.net = getattr(
            models,
            model)(num_classes=None).eval().requires_grad_(False).to(device)
        self.net.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(checkpoint), map_location=device))

        # data transforms
        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, imgs, num_workers=0):
        r"""imgs:   Either a PIL.Image or a list of PIL.Image instances.
        """
        # preprocess
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        assert isinstance(imgs,
                          (tuple, list)) and isinstance(imgs[0], Image.Image)
        imgs = torch.stack(parallel(self.transforms, imgs, num_workers), dim=0)

        # forward
        feats = []
        for batch in imgs.split(self.batch_size, dim=0):
            batch = batch.to(self.device, non_blocking=True)
            feats.append(self.net(batch))
        return torch.cat(feats, dim=0)

    def batch_process(self, paths):
        # init dataloader
        dataloader = DataLoader(
            dataset=ImageFolder(paths, self.transforms),
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=8,
            prefetch_factor=2)

        # forward
        feats = []
        for step, batch in enumerate(dataloader, 1):
            print(f'Step: {step}/{len(dataloader)}', flush=True)
            batch = batch.to(self.device, non_blocking=True)
            feats.append(self.net(batch))
        return torch.cat(feats)


class Classifier(object):

    def __init__(
        self,
        model='InceptionV1',
        checkpoint='models/classifier/shoes+apparel+bag-sgdetect-211230.pth',
        num_classes=1,
        resolution=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        batch_size=64,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.num_classes = num_classes
        self.resolution = resolution
        self.batch_size = batch_size
        self.device = device

        # init model
        self.net = getattr(models, model)(
            num_classes=num_classes).eval().requires_grad_(False).to(device)
        self.net.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(checkpoint), map_location=device))

        # data transforms
        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, imgs, num_workers=0):
        r"""imgs:   Either a PIL.Image or a list of PIL.Image instances.
        """
        # preprocess
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        assert isinstance(imgs,
                          (tuple, list)) and isinstance(imgs[0], Image.Image)
        imgs = torch.stack(parallel(self.transforms, imgs, num_workers), dim=0)

        # forward
        scores = []
        for batch in imgs.split(self.batch_size, dim=0):
            batch = batch.to(self.device, non_blocking=True)
            logits = self.net(batch)
            scores.append(logits.sigmoid() if self.num_classes ==  # noqa W504
                          1 else logits.softmax(dim=1))
        return torch.cat(scores, dim=0)


class Text2Image(object):

    def __init__(
        self,
        vqgan_dim=128,
        vqgan_z_dim=256,
        vqgan_dim_mult=[1, 1, 2, 2, 4],
        vqgan_num_res_blocks=2,
        vqgan_attn_scales=[1.0 / 16],
        vqgan_codebook_size=975,
        vqgan_beta=0.25,
        gpt_txt_vocab_size=21128,
        gpt_txt_seq_len=64,
        gpt_img_seq_len=1024,
        gpt_dim=1024,
        gpt_num_heads=16,
        gpt_num_layers=24,
        vqgan_checkpoint='models/vqgan/vqgan_shoes+apparels_step10k_vocab975.pth',
        gpt_checkpoint='models/seq2seq_gpt/text2image_shoes+apparels_step400k.pth',
        tokenizer=data.BertTokenizer(name='bert-base-chinese', length=64),
        batch_size=16,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device

        # init VQGAN model
        self.vqgan = models.VQGAN(
            dim=vqgan_dim,
            z_dim=vqgan_z_dim,
            dim_mult=vqgan_dim_mult,
            num_res_blocks=vqgan_num_res_blocks,
            attn_scales=vqgan_attn_scales,
            codebook_size=vqgan_codebook_size,
            beta=vqgan_beta).eval().requires_grad_(False).to(device)
        self.vqgan.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE(vqgan_checkpoint), map_location=device))

        # init GPT model
        self.gpt = models.Seq2SeqGPT(
            src_vocab_size=gpt_txt_vocab_size,
            tar_vocab_size=vqgan_codebook_size,
            src_seq_len=gpt_txt_seq_len,
            tar_seq_len=gpt_img_seq_len,
            dim=gpt_dim,
            num_heads=gpt_num_heads,
            num_layers=gpt_num_layers).eval().requires_grad_(False).to(device)
        self.gpt.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(gpt_checkpoint), map_location=device))

    def __call__(self,
                 txts,
                 top_k=64,
                 top_p=None,
                 temperature=0.6,
                 use_fp16=True):
        # preprocess
        if isinstance(txts, str):
            txts = [txts]
        assert isinstance(txts, (tuple, list)) and isinstance(txts[0], str)
        txt_tokens = torch.LongTensor([self.tokenizer(u) for u in txts])

        # forward
        out_imgs = []
        for batch in txt_tokens.split(self.batch_size, dim=0):
            # sample
            batch = batch.to(self.device, non_blocking=True)
            with amp.autocast(enabled=use_fp16):
                img_tokens = self.gpt.sample(batch, top_k, top_p, temperature)

            # decode
            imgs = self.vqgan.decode_from_tokens(img_tokens)
            imgs = self._whiten_borders(imgs)
            imgs = imgs.clamp_(-1, 1).add_(1).mul_(125.0).permute(
                0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            imgs = [Image.fromarray(u) for u in imgs]

            # append
            out_imgs += imgs
        return out_imgs

    def _whiten_borders(self, imgs):
        r"""Remove border artifacts.
        """
        imgs[:, :, :18, :] = 1
        imgs[:, :, :, :18] = 1
        imgs[:, :, -18:, :] = 1
        imgs[:, :, :, -18:] = 1
        return imgs


class Sole2Shoe(object):

    def __init__(
        self,
        vqgan_dim=128,
        vqgan_z_dim=256,
        vqgan_dim_mult=[1, 1, 2, 2, 4],
        vqgan_num_res_blocks=2,
        vqgan_attn_scales=[1.0 / 16],
        vqgan_codebook_size=975,
        vqgan_beta=0.25,
        src_resolution=256,
        tar_resolution=512,
        gpt_dim=1024,
        gpt_num_heads=16,
        gpt_num_layers=24,
        vqgan_checkpoint='models/vqgan/vqgan_shoes+apparels_step10k_vocab975.pth',
        gpt_checkpoint='models/seq2seq_gpt/sole2shoe-step300k-220104.pth',
        batch_size=12,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.batch_size = batch_size
        self.device = device
        src_seq_len = (src_resolution // 16)**2
        tar_seq_len = (tar_resolution // 16)**2

        # init VQGAN model
        self.vqgan = models.VQGAN(
            dim=vqgan_dim,
            z_dim=vqgan_z_dim,
            dim_mult=vqgan_dim_mult,
            num_res_blocks=vqgan_num_res_blocks,
            attn_scales=vqgan_attn_scales,
            codebook_size=vqgan_codebook_size,
            beta=vqgan_beta).eval().requires_grad_(False).to(device)
        self.vqgan.load_state_dict(
            torch.load(
                DOWNLOAD_TO_CACHE(vqgan_checkpoint), map_location=device))

        # init GPT model
        self.gpt = models.Seq2SeqGPT(
            src_vocab_size=vqgan_codebook_size,
            tar_vocab_size=vqgan_codebook_size,
            src_seq_len=src_seq_len,
            tar_seq_len=tar_seq_len,
            dim=gpt_dim,
            num_heads=gpt_num_heads,
            num_layers=gpt_num_layers).eval().requires_grad_(False).to(device)
        self.gpt.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(gpt_checkpoint), map_location=device))

        # data transforms
        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(src_resolution),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self,
                 sole_imgs,
                 top_k=64,
                 top_p=None,
                 temperature=0.6,
                 use_fp16=True,
                 num_workers=0):
        # preprocess
        if isinstance(sole_imgs, Image.Image):
            sole_imgs = [sole_imgs]
        assert isinstance(sole_imgs, (tuple, list)) and isinstance(
            sole_imgs[0], Image.Image)
        sole_imgs = torch.stack(
            parallel(self.transforms, sole_imgs, num_workers), dim=0)

        # forward
        out_imgs = []
        for batch in sole_imgs.split(self.batch_size, dim=0):
            # sample
            batch = batch.to(self.device)
            with amp.autocast(enabled=use_fp16):
                sole_tokens = self.vqgan.encode_to_tokens(batch)
                shoe_tokens = self.gpt.sample(sole_tokens, top_k, top_p,
                                              temperature)

            # decode
            shoe_imgs = self.vqgan.decode_from_tokens(shoe_tokens)
            shoe_imgs = self._whiten_borders(shoe_imgs)
            shoe_imgs = shoe_imgs.clamp_(-1, 1).add_(1).mul_(125.0).permute(
                0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            shoe_imgs = [Image.fromarray(u) for u in shoe_imgs]

            # append
            out_imgs += shoe_imgs
        return out_imgs

    def _whiten_borders(self, imgs):
        r"""Remove border artifacts.
        """
        imgs[:, :, :18, :] = 1
        imgs[:, :, :, :18] = 1
        imgs[:, :, -18:, :] = 1
        imgs[:, :, :, -18:] = 1
        return imgs


class ImageParser(object):

    def __init__(
        self,
        model='SPNet',
        num_classes=2,
        resolution=800,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        model_with_softmax=False,
        checkpoint='models/spnet/sole_segmentation_211219.pth',
        batch_size=16,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.batch_size = batch_size
        self.device = device

        # init model
        if checkpoint.endswith('.pt'):
            self.net = torch.jit.load(
                DOWNLOAD_TO_CACHE(checkpoint)).eval().to(device)
            [p.requires_grad_(False) for p in self.net.parameters()]
        else:
            self.net = getattr(models, model)(
                num_classes=num_classes,
                pretrained=False).eval().requires_grad_(False).to(device)
            self.net.load_state_dict(
                torch.load(DOWNLOAD_TO_CACHE(checkpoint), map_location=device))
        self.softmax = (lambda x, dim: x) if model_with_softmax else F.softmax

        # data transforms
        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(resolution),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, imgs, num_workers=0):
        # preprocess
        if isinstance(imgs, Image.Image):
            imgs = [imgs]
        assert isinstance(imgs,
                          (tuple, list)) and isinstance(imgs[0], Image.Image)
        sizes = [u.size for u in imgs]
        imgs = torch.stack(parallel(self.transforms, imgs, num_workers), dim=0)

        # forward
        masks = []
        for batch in imgs.split(self.batch_size, dim=0):
            batch = batch.to(self.device, non_blocking=True)
            masks.append(self.softmax(self.net(batch), dim=1))

        # postprocess
        masks = torch.cat(masks, dim=0).unsqueeze(1)
        masks = [
            F.interpolate(u, v, mode='bilinear', align_corners=False)
            for u, v in zip(masks, sizes)
        ]
        return masks


class TextImageMatch(object):

    def __init__(
        self,
        embed_dim=512,
        image_size=224,
        patch_size=32,
        vision_dim=768,
        vision_heads=12,
        vision_layers=12,
        vocab_size=21128,
        text_len=77,
        text_dim=512,
        text_heads=8,
        text_layers=12,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        checkpoint='models/clip/clip_shoes+apparels_step84k_210105.pth',
        tokenizer=data.BertTokenizer(name='bert-base-chinese', length=77),
        batch_size=64,
        device=torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')):  # noqa E125
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device

        # init model
        self.clip = models.CLIP(
            embed_dim=embed_dim,
            image_size=image_size,
            patch_size=patch_size,
            vision_dim=vision_dim,
            vision_heads=vision_heads,
            vision_layers=vision_layers,
            vocab_size=vocab_size,
            text_len=text_len,
            text_dim=text_dim,
            text_heads=text_heads,
            text_layers=text_layers).eval().requires_grad_(False).to(device)
        self.clip.load_state_dict(
            torch.load(DOWNLOAD_TO_CACHE(checkpoint), map_location=device))

        # transforms
        scale_size = int(image_size * 8 / 7)
        self.transforms = T.Compose([
            data.PadToSquare(),
            T.Resize(scale_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean, std)
        ])

    def __call__(self, imgs, txts, num_workers=0):
        # preprocess
        assert isinstance(imgs,
                          (tuple, list)) and isinstance(imgs[0], Image.Image)
        assert isinstance(txts, (tuple, list)) and isinstance(txts[0], str)
        txt_tokens = torch.LongTensor([self.tokenizer(u) for u in txts])
        imgs = torch.stack(parallel(self.transforms, imgs, num_workers), dim=0)

        # forward
        scores = []
        for img_batch, txt_batch in zip(
                imgs.split(self.batch_size, dim=0),
                txt_tokens.split(self.batch_size, dim=0)):
            img_batch = img_batch.to(self.device)
            txt_batch = txt_batch.to(self.device)
            xi = F.normalize(self.clip.visual(img_batch), p=2, dim=1)
            xt = F.normalize(self.clip.textual(txt_batch), p=2, dim=1)
            scores.append((xi * xt).sum(dim=1))
        return torch.cat(scores, dim=0)


def taobao_feature_extractor(category='shoes', **kwargs):
    r"""Pretrained taobao-search feature extractors.
    """
    assert category in ['softall', 'shoes', 'bag']
    checkpoint = osp.join(
        'models/inception-v1', {
            'softall': '1214softall_10.10.0.5000',
            'shoes': '1218shoes.v9_7.140.0.1520000',
            'bag': '0926bag.v9_6.29.0.140000'
        }[category])
    app = FeatureExtractor(
        model='InceptionV1',
        checkpoint=checkpoint,
        resolution=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        **kwargs)
    return app


def singleton_classifier(**kwargs):
    r"""Pretrained classifier that finds single-object images.
        Supports shoes, apparel, and bag images.
    """
    app = Classifier(
        model='InceptionV1',
        checkpoint='models/classifier/shoes+apparel+bag-sgdetect-211230.pth',
        num_classes=1,
        resolution=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        **kwargs)
    return app


def orientation_classifier(**kwargs):
    r"""Shoes orientation classifier.
    """
    app = Classifier(
        model='InceptionV1',
        checkpoint='models/classifier/shoes-oriendetect-20211026.pth',
        num_classes=1,
        resolution=224,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        **kwargs)
    return app


def fashion_text2image(**kwargs):
    r"""Fashion text-to-image generator.
        Supports shoe and apparel image generation.
    """
    app = Text2Image(
        vqgan_dim=128,
        vqgan_z_dim=256,
        vqgan_dim_mult=[1, 1, 2, 2, 4],
        vqgan_num_res_blocks=2,
        vqgan_attn_scales=[1.0 / 16],
        vqgan_codebook_size=975,
        vqgan_beta=0.25,
        gpt_txt_vocab_size=21128,
        gpt_txt_seq_len=64,
        gpt_img_seq_len=1024,
        gpt_dim=1024,
        gpt_num_heads=16,
        gpt_num_layers=24,
        vqgan_checkpoint=  # noqa E251
        'models/vqgan/vqgan_shoes+apparels_step10k_vocab975.pth',
        gpt_checkpoint=  # noqa E251
        'models/seq2seq_gpt/text2image_shoes+apparels_step400k.pth',
        tokenizer=data.BertTokenizer(name='bert-base-chinese', length=64),
        **kwargs)
    return app


def mindalle_text2image(**kwargs):
    r"""Pretrained text2image generator with weights copied from minDALL-E.
    """
    app = Text2Image(
        vqgan_dim=128,
        vqgan_z_dim=256,
        vqgan_dim_mult=[1, 1, 2, 2, 4],
        vqgan_num_res_blocks=2,
        vqgan_attn_scales=[1.0 / 16],
        vqgan_codebook_size=16384,
        vqgan_beta=0.25,
        gpt_txt_vocab_size=16384,
        gpt_txt_seq_len=64,
        gpt_img_seq_len=256,
        gpt_dim=1536,
        gpt_num_heads=24,
        gpt_num_layers=42,
        vqgan_checkpoint='models/minDALLE/1.3B_vqgan.pth',
        gpt_checkpoint='models/minDALLE/1.3B_gpt.pth',
        tokenizer=data.BPETokenizer(length=64),
        **kwargs)
    return app


def sole2shoe(**kwargs):
    app = Sole2Shoe(
        vqgan_dim=128,
        vqgan_z_dim=256,
        vqgan_dim_mult=[1, 1, 2, 2, 4],
        vqgan_num_res_blocks=2,
        vqgan_attn_scales=[1.0 / 16],
        vqgan_codebook_size=975,
        vqgan_beta=0.25,
        src_resolution=256,
        tar_resolution=512,
        gpt_dim=1024,
        gpt_num_heads=16,
        gpt_num_layers=24,
        vqgan_checkpoint=  # noqa E251
        'models/vqgan/vqgan_shoes+apparels_step10k_vocab975.pth',
        gpt_checkpoint='models/seq2seq_gpt/sole2shoe-step300k-220104.pth',
        **kwargs)
    return app


def sole_parser(**kwargs):
    app = ImageParser(
        model='SPNet',
        num_classes=2,
        resolution=800,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        model_with_softmax=False,
        checkpoint='models/spnet/sole_segmentation_211219.pth',
        **kwargs)
    return app


def sod_foreground_parser(**kwargs):
    app = ImageParser(
        model=None,
        num_classes=None,
        resolution=448,
        mean=[0.488431, 0.466275, 0.403686],
        std=[0.222627, 0.21949, 0.22549],
        model_with_softmax=True,
        checkpoint='models/semseg/sod_model_20201228.pt',
        **kwargs)
    return app


def fashion_text_image_match(**kwargs):
    app = TextImageMatch(
        embed_dim=512,
        image_size=224,
        patch_size=32,
        vision_dim=768,
        vision_heads=12,
        vision_layers=12,
        vocab_size=21128,
        text_len=77,
        text_dim=512,
        text_heads=8,
        text_layers=12,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        checkpoint='models/clip/clip_shoes+apparels_step84k_210105.pth',
        tokenizer=data.BertTokenizer(name='bert-base-chinese', length=77),
        **kwargs)
    return app
