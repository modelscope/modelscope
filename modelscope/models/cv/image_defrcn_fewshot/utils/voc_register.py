# The implementation is adopted from er-muyue/DeFRCN
# made publicly available under the MIT License at
# https://github.com/er-muyue/DeFRCN/blob/main/defrcn/data/meta_voc.py

import os
import xml.etree.ElementTree as ET

import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager

# PASCAL VOC categories
PASCAL_VOC_ALL_CATEGORIES = {
    1: [
        'aeroplane',
        'bicycle',
        'boat',
        'bottle',
        'car',
        'cat',
        'chair',
        'diningtable',
        'dog',
        'horse',
        'person',
        'pottedplant',
        'sheep',
        'train',
        'tvmonitor',
        'bird',
        'bus',
        'cow',
        'motorbike',
        'sofa',
    ],
    2: [
        'bicycle',
        'bird',
        'boat',
        'bus',
        'car',
        'cat',
        'chair',
        'diningtable',
        'dog',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'train',
        'tvmonitor',
        'aeroplane',
        'bottle',
        'cow',
        'horse',
        'sofa',
    ],
    3: [
        'aeroplane',
        'bicycle',
        'bird',
        'bottle',
        'bus',
        'car',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'person',
        'pottedplant',
        'train',
        'tvmonitor',
        'boat',
        'cat',
        'motorbike',
        'sheep',
        'sofa',
    ]
}

PASCAL_VOC_NOVEL_CATEGORIES = {
    1: ['bird', 'bus', 'cow', 'motorbike', 'sofa'],
    2: ['aeroplane', 'bottle', 'cow', 'horse', 'sofa'],
    3: ['boat', 'cat', 'motorbike', 'sheep', 'sofa']
}

PASCAL_VOC_BASE_CATEGORIES = {
    1: [
        'aeroplane',
        'bicycle',
        'boat',
        'bottle',
        'car',
        'cat',
        'chair',
        'diningtable',
        'dog',
        'horse',
        'person',
        'pottedplant',
        'sheep',
        'train',
        'tvmonitor',
    ],
    2: [
        'bicycle',
        'bird',
        'boat',
        'bus',
        'car',
        'cat',
        'chair',
        'diningtable',
        'dog',
        'motorbike',
        'person',
        'pottedplant',
        'sheep',
        'train',
        'tvmonitor',
    ],
    3: [
        'aeroplane',
        'bicycle',
        'bird',
        'bottle',
        'bus',
        'car',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'person',
        'pottedplant',
        'train',
        'tvmonitor',
    ]
}


def load_filtered_voc_instances(name: str, root: str, dirname: str, split: str,
                                classnames: str):
    """
    Load Pascal VOC detection annotations to Detectron2 format.
    Args:
        dirname: Contain "Annotations", "ImageSets", "JPEGImages"
        split (str): one of "train", "test", "val", "trainval"
    """
    is_shots = 'shot' in name
    dicts = []
    if is_shots:
        fileids = {}
        # split_dir = os.path.join("datasets", "vocsplit")
        split_dir = os.path.join(root, 'vocsplit')
        shot = name.split('_')[-2].split('shot')[0]
        seed = int(name.split('_seed')[-1])
        split_dir = os.path.join(split_dir, 'seed{}'.format(seed))
        for cls in classnames:
            with PathManager.open(
                    os.path.join(split_dir,
                                 'box_{}shot_{}_train.txt'.format(shot,
                                                                  cls))) as f:
                fileids_ = np.loadtxt(f, dtype=np.str_).tolist()
                if isinstance(fileids_, str):
                    fileids_ = [fileids_]
                fileids_ = [
                    fid.split('/')[-1].split('.jpg')[0] for fid in fileids_
                ]
                fileids[cls] = fileids_

        for cls, fileids_ in fileids.items():
            dicts_ = []
            for fileid in fileids_:
                year = '2012' if '_' in fileid else '2007'
                # dirname = os.path.join("datasets", "VOC{}".format(year))
                # anno_file = os.path.join(dirname, "Annotations", fileid + ".xml")
                # jpeg_file = os.path.join(dirname, "JPEGImages", fileid + ".jpg")

                dir_voc = os.path.join(root, 'VOC{}'.format(year))
                anno_file = os.path.join(dir_voc, 'Annotations',
                                         fileid + '.xml')
                jpeg_file = os.path.join(dir_voc, 'JPEGImages',
                                         fileid + '.jpg')

                tree = ET.parse(anno_file)

                for obj in tree.findall('object'):
                    r = {
                        'file_name': jpeg_file,
                        'image_id': fileid,
                        'height': int(tree.findall('./size/height')[0].text),
                        'width': int(tree.findall('./size/width')[0].text),
                    }
                    cls_ = obj.find('name').text
                    if cls != cls_:
                        continue
                    bbox = obj.find('bndbox')
                    bbox = [
                        float(bbox.find(x).text)
                        for x in ['xmin', 'ymin', 'xmax', 'ymax']
                    ]
                    bbox[0] -= 1.0
                    bbox[1] -= 1.0

                    instances = [{
                        'category_id': classnames.index(cls),
                        'bbox': bbox,
                        'bbox_mode': BoxMode.XYXY_ABS,
                    }]
                    r['annotations'] = instances
                    dicts_.append(r)
            if len(dicts_) > int(shot):
                dicts_ = np.random.choice(dicts_, int(shot), replace=False)
            dicts.extend(dicts_)
    else:
        with PathManager.open(
                os.path.join(root, dirname, 'ImageSets', 'Main',
                             split + '.txt')) as f:
            fileids = np.loadtxt(f, dtype=np.str_)

        for fileid in fileids:
            anno_file = os.path.join(root, dirname, 'Annotations',
                                     fileid + '.xml')
            jpeg_file = os.path.join(root, dirname, 'JPEGImages',
                                     fileid + '.jpg')

            tree = ET.parse(anno_file)

            r = {
                'file_name': jpeg_file,
                'image_id': fileid,
                'height': int(tree.findall('./size/height')[0].text),
                'width': int(tree.findall('./size/width')[0].text),
            }
            instances = []

            for obj in tree.findall('object'):
                cls = obj.find('name').text
                if not (cls in classnames):
                    continue
                bbox = obj.find('bndbox')
                bbox = [
                    float(bbox.find(x).text)
                    for x in ['xmin', 'ymin', 'xmax', 'ymax']
                ]
                bbox[0] -= 1.0
                bbox[1] -= 1.0

                instances.append({
                    'category_id': classnames.index(cls),
                    'bbox': bbox,
                    'bbox_mode': BoxMode.XYXY_ABS,
                })
            r['annotations'] = instances
            dicts.append(r)

    return dicts


def register_meta_voc(name, root, dirname, split, year, keepclasses, sid):
    if keepclasses.startswith('base_novel'):
        thing_classes = PASCAL_VOC_ALL_CATEGORIES[sid]
    elif keepclasses.startswith('base'):
        thing_classes = PASCAL_VOC_BASE_CATEGORIES[sid]
    elif keepclasses.startswith('novel'):
        thing_classes = PASCAL_VOC_NOVEL_CATEGORIES[sid]

    DatasetCatalog.register(
        name,
        lambda: load_filtered_voc_instances(name, root, dirname, split,
                                            thing_classes),
    )

    MetadataCatalog.get(name).set(
        thing_classes=thing_classes,
        dirname=os.path.join(root, dirname),
        year=year,
        split=split,
        base_classes=PASCAL_VOC_BASE_CATEGORIES[sid],
        novel_classes=PASCAL_VOC_NOVEL_CATEGORIES[sid],
    )


def register_all_voc(root='datasets'):

    METASPLITS = [
        ('voc_2007_trainval_base1', 'VOC2007', 'trainval', 'base1', 1),
        ('voc_2007_trainval_base2', 'VOC2007', 'trainval', 'base2', 2),
        ('voc_2007_trainval_base3', 'VOC2007', 'trainval', 'base3', 3),
        ('voc_2012_trainval_base1', 'VOC2012', 'trainval', 'base1', 1),
        ('voc_2012_trainval_base2', 'VOC2012', 'trainval', 'base2', 2),
        ('voc_2012_trainval_base3', 'VOC2012', 'trainval', 'base3', 3),
        ('voc_2007_trainval_all1', 'VOC2007', 'trainval', 'base_novel_1', 1),
        ('voc_2007_trainval_all2', 'VOC2007', 'trainval', 'base_novel_2', 2),
        ('voc_2007_trainval_all3', 'VOC2007', 'trainval', 'base_novel_3', 3),
        ('voc_2012_trainval_all1', 'VOC2012', 'trainval', 'base_novel_1', 1),
        ('voc_2012_trainval_all2', 'VOC2012', 'trainval', 'base_novel_2', 2),
        ('voc_2012_trainval_all3', 'VOC2012', 'trainval', 'base_novel_3', 3),
        ('voc_2007_test_base1', 'VOC2007', 'test', 'base1', 1),
        ('voc_2007_test_base2', 'VOC2007', 'test', 'base2', 2),
        ('voc_2007_test_base3', 'VOC2007', 'test', 'base3', 3),
        ('voc_2007_test_novel1', 'VOC2007', 'test', 'novel1', 1),
        ('voc_2007_test_novel2', 'VOC2007', 'test', 'novel2', 2),
        ('voc_2007_test_novel3', 'VOC2007', 'test', 'novel3', 3),
        ('voc_2007_test_all1', 'VOC2007', 'test', 'base_novel_1', 1),
        ('voc_2007_test_all2', 'VOC2007', 'test', 'base_novel_2', 2),
        ('voc_2007_test_all3', 'VOC2007', 'test', 'base_novel_3', 3),
    ]
    for prefix in ['all', 'novel']:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = '_seed{}'.format(seed)
                        name = 'voc_{}_trainval_{}{}_{}shot{}'.format(
                            year, prefix, sid, shot, seed)
                        dirname = 'VOC{}'.format(year)
                        img_file = '{}_{}shot_split_{}_trainval'.format(
                            prefix, shot, sid)
                        keepclasses = ('base_novel_{}'.format(sid) if prefix
                                       == 'all' else 'novel{}'.format(sid))
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid))

    for name, dirname, split, keepclasses, sid in METASPLITS:
        if name in DatasetCatalog:
            continue

        year = 2007 if '2007' in name else 2012
        register_meta_voc(
            name,
            root,
            dirname,
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = 'pascal_voc'
