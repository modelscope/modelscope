import webdataset as wds
import kornia
from PIL import Image
import io
import os
import torchvision
from PIL import Image
import glob
import random
import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import rearrange
import torch
from webdataset.handlers import warn_and_continue


from ldm.util import instantiate_from_config
from ldm.data.inpainting.synthetic_mask import gen_large_mask, MASK_MODES
from ldm.data.base import PRNGMixin


class DataWithWings(torch.utils.data.IterableDataset):
    def __init__(self, min_size, transform=None, target_transform=None):
        self.min_size = min_size
        self.transform = transform if transform is not None else nn.Identity()
        self.target_transform = target_transform if target_transform is not None else nn.Identity()
        self.kv = OnDiskKV(file='/home/ubuntu/laion5B-watermark-safety-ordered', key_format='q', value_format='ee')
        self.kv_aesthetic = OnDiskKV(file='/home/ubuntu/laion5B-aesthetic-tags-kv', key_format='q', value_format='e')
        self.pwatermark_threshold = 0.8
        self.punsafe_threshold = 0.5
        self.aesthetic_threshold = 5.
        self.total_samples = 0
        self.samples = 0
        location = 'pipe:aws s3 cp --quiet s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -'

        self.inner_dataset = wds.DataPipeline(
            wds.ResampledShards(location),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(1000, handler=wds.warn_and_continue),
            wds.decode('pilrgb', handler=wds.warn_and_continue),
            wds.map(self._add_tags, handler=wds.ignore_and_continue),
            wds.select(self._filter_predicate),
            wds.map_dict(jpg=self.transform, txt=self.target_transform, punsafe=self._punsafe_to_class, handler=wds.warn_and_continue),
            wds.to_tuple('jpg', 'txt', 'punsafe', handler=wds.warn_and_continue),
        )

    @staticmethod
    def _compute_hash(url, text):
        if url is None:
            url = ''
        if text is None:
            text = ''
        total = (url + text).encode('utf-8')
        return mmh3.hash64(total)[0]

    def _add_tags(self, x):
        hsh = self._compute_hash(x['json']['url'], x['txt'])
        pwatermark, punsafe = self.kv[hsh]
        aesthetic = self.kv_aesthetic[hsh][0]
        return {**x, 'pwatermark': pwatermark, 'punsafe': punsafe, 'aesthetic': aesthetic}

    def _punsafe_to_class(self, punsafe):
        return torch.tensor(punsafe >= self.punsafe_threshold).long()

    def _filter_predicate(self, x):
        try:
            return x['pwatermark'] < self.pwatermark_threshold and x['aesthetic'] >= self.aesthetic_threshold and x['json']['original_width'] >= self.min_size and x['json']['original_height'] >= self.min_size
        except:
            return False

    def __iter__(self):
        return iter(self.inner_dataset)


def dict_collation_fn(samples, combine_tensors=True, combine_scalars=True):
    """Take a list  of samples (as dictionary) and create a batch, preserving the keys.
    If `tensors` is True, `ndarray` objects are combined into
    tensor batches.
    :param dict samples: list of samples
    :param bool tensors: whether to turn lists of ndarrays into a single ndarray
    :returns: single sample consisting of a batch
    :rtype: dict
    """
    keys = set.intersection(*[set(sample.keys()) for sample in samples])
    batched = {key: [] for key in keys}

    for s in samples:
        [batched[key].append(s[key]) for key in batched]

    result = {}
    for key in batched:
        if isinstance(batched[key][0], (int, float)):
            if combine_scalars:
                result[key] = np.array(list(batched[key]))
        elif isinstance(batched[key][0], torch.Tensor):
            if combine_tensors:
                result[key] = torch.stack(list(batched[key]))
        elif isinstance(batched[key][0], np.ndarray):
            if combine_tensors:
                result[key] = np.array(list(batched[key]))
        else:
            result[key] = list(batched[key])
    return result


class WebDataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, tar_base, batch_size, train=None, validation=None,
                 test=None, num_workers=4, multinode=True, min_size=None,
                 max_pwatermark=1.0,
                 **kwargs):
        super().__init__(self)
        print(f'Setting tar base to {tar_base}')
        self.tar_base = tar_base
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train = train
        self.validation = validation
        self.test = test
        self.multinode = multinode
        self.min_size = min_size  # filter out very small images
        self.max_pwatermark = max_pwatermark # filter out watermarked images

    def make_loader(self, dataset_config, train=True):
        if 'image_transforms' in dataset_config:
            image_transforms = [instantiate_from_config(tt) for tt in dataset_config.image_transforms]
        else:
            image_transforms = []

        image_transforms.extend([torchvision.transforms.ToTensor(),
                                 torchvision.transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = torchvision.transforms.Compose(image_transforms)

        if 'transforms' in dataset_config:
            transforms_config = OmegaConf.to_container(dataset_config.transforms)
        else:
            transforms_config = dict()

        transform_dict = {dkey: load_partial_from_config(transforms_config[dkey])
                if transforms_config[dkey] != 'identity' else identity
                for dkey in transforms_config}
        img_key = dataset_config.get('image_key', 'jpeg')
        transform_dict.update({img_key: image_transforms})

        if 'postprocess' in dataset_config:
            postprocess = instantiate_from_config(dataset_config['postprocess'])
        else:
            postprocess = None

        shuffle = dataset_config.get('shuffle', 0)
        shardshuffle = shuffle > 0

        nodesplitter = wds.shardlists.split_by_node if self.multinode else wds.shardlists.single_node_only

        if self.tar_base == "__improvedaesthetic__":
            print("## Warning, loading the same improved aesthetic dataset "
                    "for all splits and ignoring shards parameter.")
            tars = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
        else:
            tars = os.path.join(self.tar_base, dataset_config.shards)

        dset = wds.WebDataset(
                tars,
                nodesplitter=nodesplitter,
                shardshuffle=shardshuffle,
                handler=wds.warn_and_continue).repeat().shuffle(shuffle)
        print(f'Loading webdataset with {len(dset.pipeline[0].urls)} shards.')

        dset = (dset
                .select(self.filter_keys)
                .decode('pil', handler=wds.warn_and_continue)
                .select(self.filter_size)
                .map_dict(**transform_dict, handler=wds.warn_and_continue)
                )
        if postprocess is not None:
            dset = dset.map(postprocess)
        dset = (dset
                .batched(self.batch_size, partial=False,
                    collation_fn=dict_collation_fn)
                )

        loader = wds.WebLoader(dset, batch_size=None, shuffle=False,
                               num_workers=self.num_workers)

        return loader

    def filter_size(self, x):
        try:
            valid = True
            if self.min_size is not None and self.min_size > 1:
                try:
                    valid = valid and x['json']['original_width'] >= self.min_size and x['json']['original_height'] >= self.min_size
                except Exception:
                    valid = False
            if self.max_pwatermark is not None and self.max_pwatermark < 1.0:
                try:
                    valid = valid and  x['json']['pwatermark'] <= self.max_pwatermark
                except Exception:
                    valid = False
            return valid
        except Exception:
            return False

    def filter_keys(self, x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def train_dataloader(self):
        return self.make_loader(self.train)

    def val_dataloader(self):
        return self.make_loader(self.validation, train=False)

    def test_dataloader(self):
        return self.make_loader(self.test, train=False)


from ldm.modules.image_degradation import degradation_fn_bsr_light
import cv2

class AddLR(object):
    def __init__(self, factor, output_size, initial_size=None, image_key="jpg"):
        self.factor = factor
        self.output_size = output_size
        self.image_key = image_key
        self.initial_size = initial_size

    def pt2np(self, x):
        x = ((x+1.0)*127.5).clamp(0, 255).to(dtype=torch.uint8).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x)/127.5-1.0
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = self.pt2np(sample[self.image_key])
        if self.initial_size is not None:
            x = cv2.resize(x, (self.initial_size, self.initial_size), interpolation=2)
        x = degradation_fn_bsr_light(x, sf=self.factor)['image']
        x = cv2.resize(x, (self.output_size, self.output_size), interpolation=2)
        x = self.np2pt(x)
        sample['lr'] = x
        return sample

class AddBW(object):
    def __init__(self, image_key="jpg"):
        self.image_key = image_key

    def pt2np(self, x):
        x = ((x+1.0)*127.5).clamp(0, 255).to(dtype=torch.uint8).detach().cpu().numpy()
        return x

    def np2pt(self, x):
        x = torch.from_numpy(x)/127.5-1.0
        return x

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = sample[self.image_key]
        w = torch.rand(3, device=x.device)
        w /= w.sum()
        out = torch.einsum('hwc,c->hw', x, w)

        # Keep as 3ch so we can pass to encoder, also we might want to add hints
        sample['lr'] = out.unsqueeze(-1).tile(1,1,3)
        return sample

class AddMask(PRNGMixin):
    def __init__(self, mode="512train", p_drop=0.):
        super().__init__()
        assert mode in list(MASK_MODES.keys()), f'unknown mask generation mode "{mode}"'
        self.make_mask = MASK_MODES[mode]
        self.p_drop = p_drop

    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = sample['jpg']
        mask = self.make_mask(self.prng, x.shape[0], x.shape[1])
        if self.prng.choice(2, p=[1 - self.p_drop, self.p_drop]):
            mask = np.ones_like(mask)
        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1
        mask = torch.from_numpy(mask[..., None])
        sample['mask'] = mask
        sample['masked_image'] = x * (mask < 0.5)
        return sample


class AddEdge(PRNGMixin):
    def __init__(self, mode="512train", mask_edges=True):
        super().__init__()
        assert mode in list(MASK_MODES.keys()), f'unknown mask generation mode "{mode}"'
        self.make_mask = MASK_MODES[mode]
        self.n_down_choices = [0]
        self.sigma_choices = [1, 2]
        self.mask_edges = mask_edges

    @torch.no_grad()
    def __call__(self, sample):
        # sample['jpg'] is tensor hwc in [-1, 1] at this point
        x = sample['jpg']

        mask = self.make_mask(self.prng, x.shape[0], x.shape[1])
        mask[mask < 0.5] = 0
        mask[mask > 0.5] = 1
        mask = torch.from_numpy(mask[..., None])
        sample['mask'] = mask

        n_down_idx = self.prng.choice(len(self.n_down_choices))
        sigma_idx = self.prng.choice(len(self.sigma_choices))

        n_choices = len(self.n_down_choices)*len(self.sigma_choices)
        raveled_idx = np.ravel_multi_index((n_down_idx, sigma_idx),
                                           (len(self.n_down_choices), len(self.sigma_choices)))
        normalized_idx = raveled_idx/max(1, n_choices-1)

        n_down = self.n_down_choices[n_down_idx]
        sigma = self.sigma_choices[sigma_idx]

        kernel_size = 4*sigma+1
        kernel_size = (kernel_size, kernel_size)
        sigma = (sigma, sigma)
        canny = kornia.filters.Canny(
                low_threshold=0.1,
                high_threshold=0.2,
                kernel_size=kernel_size,
                sigma=sigma,
                hysteresis=True,
                )
        y = (x+1.0)/2.0 # in 01
        y = y.unsqueeze(0).permute(0, 3, 1, 2).contiguous()

        # down
        for i_down in range(n_down):
            size = min(y.shape[-2], y.shape[-1])//2
            y = kornia.geometry.transform.resize(y, size, antialias=True)

        # edge
        _, y = canny(y)

        if n_down > 0:
            size = x.shape[0], x.shape[1]
            y = kornia.geometry.transform.resize(y, size, interpolation="nearest")

        y = y.permute(0, 2, 3, 1)[0].expand(-1, -1, 3).contiguous()
        y = y*2.0-1.0

        if self.mask_edges:
            sample['masked_image'] = y * (mask < 0.5)
        else:
            sample['masked_image'] = y
            sample['mask'] = torch.zeros_like(sample['mask'])

        # concat normalized idx
        sample['smoothing_strength'] = torch.ones_like(sample['mask'])*normalized_idx

        return sample


def example00():
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/000000.tar -"
    dataset = wds.WebDataset(url)
    example = next(iter(dataset))
    for k in example:
        print(k, type(example[k]))

    print(example["__key__"])
    for k in ["json", "txt"]:
        print(example[k].decode())

    image = Image.open(io.BytesIO(example["jpg"]))
    outdir = "tmp"
    os.makedirs(outdir, exist_ok=True)
    image.save(os.path.join(outdir, example["__key__"] + ".png"))


    def load_example(example):
        return {
            "key": example["__key__"],
            "image": Image.open(io.BytesIO(example["jpg"])),
            "text": example["txt"].decode(),
        }


    for i, example in tqdm(enumerate(dataset)):
        ex = load_example(example)
        print(ex["image"].size, ex["text"])
        if i >= 100:
            break


def example01():
    # the first laion shards contain ~10k examples each
    url = "pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..000002}.tar -"

    batch_size = 3
    shuffle_buffer = 10000
    dset = wds.WebDataset(
            url,
            nodesplitter=wds.shardlists.split_by_node,
            shardshuffle=True,
            )
    dset = (dset
            .shuffle(shuffle_buffer, initial=shuffle_buffer)
            .decode('pil', handler=warn_and_continue)
            .batched(batch_size, partial=False,
                collation_fn=dict_collation_fn)
            )

    num_workers = 2
    loader = wds.WebLoader(dset, batch_size=None, shuffle=False, num_workers=num_workers)

    batch_sizes = list()
    keys_per_epoch = list()
    for epoch in range(5):
        keys = list()
        for batch in tqdm(loader):
            batch_sizes.append(len(batch["__key__"]))
            keys.append(batch["__key__"])

        for bs in batch_sizes:
            assert bs==batch_size
        print(f"{len(batch_sizes)} batches of size {batch_size}.")
        batch_sizes = list()

        keys_per_epoch.append(keys)
        for i_batch in [0, 1, -1]:
            print(f"Batch {i_batch} of epoch {epoch}:")
            print(keys[i_batch])
        print("next epoch.")


def example02():
    from omegaconf import OmegaConf
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data import IterableDataset
    from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler
    from pytorch_lightning.trainer.supporters import CombinedLoader, CycleIterator

    #config = OmegaConf.load("configs/stable-diffusion/txt2img-1p4B-multinode-clip-encoder-high-res-512.yaml")
    #config = OmegaConf.load("configs/stable-diffusion/txt2img-upscale-clip-encoder-f16-1024.yaml")
    config = OmegaConf.load("configs/stable-diffusion/txt2img-v2-clip-encoder-improved_aesthetics-256.yaml")
    datamod = WebDataModuleFromConfig(**config["data"]["params"])
    dataloader = datamod.train_dataloader()

    for batch in dataloader:
        print(batch.keys())
        print(batch["jpg"].shape)
        break


def example03():
    # improved aesthetics
    tars = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{000000..060207}.tar -"
    dataset = wds.WebDataset(tars)

    def filter_keys(x):
        try:
            return ("jpg" in x) and ("txt" in x)
        except Exception:
            return False

    def filter_size(x):
        try:
            return x['json']['original_width'] >= 512 and x['json']['original_height'] >= 512
        except Exception:
            return False

    def filter_watermark(x):
        try:
            return x['json']['pwatermark'] < 0.5
        except Exception:
            return False

    dataset = (dataset
                .select(filter_keys)
                .decode('pil', handler=wds.warn_and_continue))
    n_save = 20
    n_total = 0
    n_large = 0
    n_large_nowm = 0
    for i, example in enumerate(dataset):
        n_total += 1
        if filter_size(example):
            n_large += 1
            if filter_watermark(example):
                n_large_nowm += 1
                if n_large_nowm < n_save+1:
                    image = example["jpg"]
                    image.save(os.path.join("tmp", f"{n_large_nowm-1:06}.png"))

        if i%500 == 0:
            print(i)
            print(f"Large: {n_large}/{n_total} | {n_large/n_total*100:.2f}%")
            if n_large > 0:
                print(f"No Watermark: {n_large_nowm}/{n_large} | {n_large_nowm/n_large*100:.2f}%")



def example04():
    # improved aesthetics
    for i_shard in range(60208)[::-1]:
        print(i_shard)
        tars = "pipe:aws s3 cp s3://s-laion/improved-aesthetics-laion-2B-en-subsets/aesthetics_tars/{:06}.tar -".format(i_shard)
        dataset = wds.WebDataset(tars)

        def filter_keys(x):
            try:
                return ("jpg" in x) and ("txt" in x)
            except Exception:
                return False

        def filter_size(x):
            try:
                return x['json']['original_width'] >= 512 and x['json']['original_height'] >= 512
            except Exception:
                return False

        dataset = (dataset
                    .select(filter_keys)
                    .decode('pil', handler=wds.warn_and_continue))
        try:
            example = next(iter(dataset))
        except Exception:
            print(f"Error @ {i_shard}")


if __name__ == "__main__":
    #example01()
    #example02()
    example03()
    #example04()
