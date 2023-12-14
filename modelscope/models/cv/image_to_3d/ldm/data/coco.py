import os
import json
import albumentations
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset
from abc import abstractmethod


class CocoBase(Dataset):
    """needed for (image, caption, segmentation) pairs"""
    def __init__(self, size=None, dataroot="", datajson="", onehot_segmentation=False, use_stuffthing=False,
                 crop_size=None, force_no_crop=False, given_files=None, use_segmentation=True,crop_type=None):
        self.split = self.get_split()
        self.size = size
        if crop_size is None:
            self.crop_size = size
        else:
            self.crop_size = crop_size

        assert crop_type in [None, 'random', 'center']
        self.crop_type = crop_type
        self.use_segmenation = use_segmentation
        self.onehot = onehot_segmentation       # return segmentation as rgb or one hot
        self.stuffthing = use_stuffthing        # include thing in segmentation
        if self.onehot and not self.stuffthing:
            raise NotImplemented("One hot mode is only supported for the "
                                 "stuffthings version because labels are stored "
                                 "a bit different.")

        data_json = datajson
        with open(data_json) as json_file:
            self.json_data = json.load(json_file)
            self.img_id_to_captions = dict()
            self.img_id_to_filepath = dict()
            self.img_id_to_segmentation_filepath = dict()

        assert data_json.split("/")[-1] in [f"captions_train{self.year()}.json",
                                            f"captions_val{self.year()}.json"]
        # TODO currently hardcoded paths, would be better to follow logic in
        # cocstuff pixelmaps
        if self.use_segmenation:
            if self.stuffthing:
                self.segmentation_prefix = (
                    f"data/cocostuffthings/val{self.year()}" if
                    data_json.endswith(f"captions_val{self.year()}.json") else
                    f"data/cocostuffthings/train{self.year()}")
            else:
                self.segmentation_prefix = (
                    f"data/coco/annotations/stuff_val{self.year()}_pixelmaps" if
                    data_json.endswith(f"captions_val{self.year()}.json") else
                    f"data/coco/annotations/stuff_train{self.year()}_pixelmaps")

        imagedirs = self.json_data["images"]
        self.labels = {"image_ids": list()}
        for imgdir in tqdm(imagedirs, desc="ImgToPath"):
            self.img_id_to_filepath[imgdir["id"]] = os.path.join(dataroot, imgdir["file_name"])
            self.img_id_to_captions[imgdir["id"]] = list()
            pngfilename = imgdir["file_name"].replace("jpg", "png")
            if self.use_segmenation:
                self.img_id_to_segmentation_filepath[imgdir["id"]] = os.path.join(
                    self.segmentation_prefix, pngfilename)
            if given_files is not None:
                if pngfilename in given_files:
                    self.labels["image_ids"].append(imgdir["id"])
            else:
                self.labels["image_ids"].append(imgdir["id"])

        capdirs = self.json_data["annotations"]
        for capdir in tqdm(capdirs, desc="ImgToCaptions"):
            # there are in average 5 captions per image
            #self.img_id_to_captions[capdir["image_id"]].append(np.array([capdir["caption"]]))
            self.img_id_to_captions[capdir["image_id"]].append(capdir["caption"])

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        if self.split=="validation":
            self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        else:
            # default option for train is random crop
            if self.crop_type in [None, 'random']:
                self.cropper = albumentations.RandomCrop(height=self.crop_size, width=self.crop_size)
            else:
                self.cropper = albumentations.CenterCrop(height=self.crop_size, width=self.crop_size)
        self.preprocessor = albumentations.Compose(
            [self.rescaler, self.cropper],
            additional_targets={"segmentation": "image"})
        if force_no_crop:
            self.rescaler = albumentations.Resize(height=self.size, width=self.size)
            self.preprocessor = albumentations.Compose(
                [self.rescaler],
                additional_targets={"segmentation": "image"})

    @abstractmethod
    def year(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.labels["image_ids"])

    def preprocess_image(self, image_path, segmentation_path=None):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        if segmentation_path:
            segmentation = Image.open(segmentation_path)
            if not self.onehot and not segmentation.mode == "RGB":
                segmentation = segmentation.convert("RGB")
            segmentation = np.array(segmentation).astype(np.uint8)
            if self.onehot:
                assert self.stuffthing
                # stored in caffe format: unlabeled==255. stuff and thing from
                # 0-181. to be compatible with the labels in
                # https://github.com/nightrome/cocostuff/blob/master/labels.txt
                # we shift stuffthing one to the right and put unlabeled in zero
                # as long as segmentation is uint8 shifting to right handles the
                # latter too
                assert segmentation.dtype == np.uint8
                segmentation = segmentation + 1

            processed = self.preprocessor(image=image, segmentation=segmentation)

            image, segmentation = processed["image"], processed["segmentation"]
        else:
            image = self.preprocessor(image=image,)['image']

        image = (image / 127.5 - 1.0).astype(np.float32)
        if segmentation_path:
            if self.onehot:
                assert segmentation.dtype == np.uint8
                # make it one hot
                n_labels = 183
                flatseg = np.ravel(segmentation)
                onehot = np.zeros((flatseg.size, n_labels), dtype=np.bool)
                onehot[np.arange(flatseg.size), flatseg] = True
                onehot = onehot.reshape(segmentation.shape + (n_labels,)).astype(int)
                segmentation = onehot
            else:
                segmentation = (segmentation / 127.5 - 1.0).astype(np.float32)
            return image, segmentation
        else:
            return image

    def __getitem__(self, i):
        img_path = self.img_id_to_filepath[self.labels["image_ids"][i]]
        if self.use_segmenation:
            seg_path = self.img_id_to_segmentation_filepath[self.labels["image_ids"][i]]
            image, segmentation = self.preprocess_image(img_path, seg_path)
        else:
            image = self.preprocess_image(img_path)
        captions = self.img_id_to_captions[self.labels["image_ids"][i]]
        # randomly draw one of all available captions per image
        caption = captions[np.random.randint(0, len(captions))]
        example = {"image": image,
                   #"caption": [str(caption[0])],
                   "caption": caption,
                   "img_path": img_path,
                   "filename_": img_path.split(os.sep)[-1]
                    }
        if self.use_segmenation:
            example.update({"seg_path": seg_path, 'segmentation': segmentation})
        return example


class CocoImagesAndCaptionsTrain2017(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,):
        super().__init__(size=size,
                         dataroot="data/coco/train2017",
                         datajson="data/coco/annotations/captions_train2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop)

    def get_split(self):
        return "train"

    def year(self):
        return '2017'


class CocoImagesAndCaptionsValidation2017(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,
                 given_files=None):
        super().__init__(size=size,
                         dataroot="data/coco/val2017",
                         datajson="data/coco/annotations/captions_val2017.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop,
                         given_files=given_files)

    def get_split(self):
        return "validation"

    def year(self):
        return '2017'



class CocoImagesAndCaptionsTrain2014(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,crop_type='random'):
        super().__init__(size=size,
                         dataroot="data/coco/train2014",
                         datajson="data/coco/annotations2014/annotations/captions_train2014.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop,
                         use_segmentation=False,
                         crop_type=crop_type)

    def get_split(self):
        return "train"

    def year(self):
        return '2014'

class CocoImagesAndCaptionsValidation2014(CocoBase):
    """returns a pair of (image, caption)"""
    def __init__(self, size, onehot_segmentation=False, use_stuffthing=False, crop_size=None, force_no_crop=False,
                 given_files=None,crop_type='center',**kwargs):
        super().__init__(size=size,
                         dataroot="data/coco/val2014",
                         datajson="data/coco/annotations2014/annotations/captions_val2014.json",
                         onehot_segmentation=onehot_segmentation,
                         use_stuffthing=use_stuffthing, crop_size=crop_size, force_no_crop=force_no_crop,
                         given_files=given_files,
                         use_segmentation=False,
                         crop_type=crop_type)

    def get_split(self):
        return "validation"

    def year(self):
        return '2014'

if __name__ == '__main__':
    with open("data/coco/annotations2014/annotations/captions_val2014.json", "r") as json_file:
        json_data = json.load(json_file)
        capdirs = json_data["annotations"]
        import pudb; pudb.set_trace()
    #d2 = CocoImagesAndCaptionsTrain2014(size=256)
    d2 = CocoImagesAndCaptionsValidation2014(size=256)
    print("constructed dataset.")
    print(f"length of {d2.__class__.__name__}: {len(d2)}")

    ex2 = d2[0]
    # ex3 = d3[0]
    # print(ex1["image"].shape)
    print(ex2["image"].shape)
    # print(ex3["image"].shape)
    # print(ex1["segmentation"].shape)
    print(ex2["caption"].__class__.__name__)
