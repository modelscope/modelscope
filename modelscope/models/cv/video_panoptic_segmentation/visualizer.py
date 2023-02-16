# The implementation is adopted from Video-K-Net,
# made publicly available at https://github.com/lxtGH/Video-K-Net

import hashlib

import cv2
import numpy as np

vip_seg_label = {
    'wall': '1',
    'ceiling': '2',
    'door': '3',
    'stair': '4',
    'ladder': '5',
    'escalator': '6',
    'Playground_slide': '7',
    'handrail_or_fence': '8',
    'window': '9',
    'others': '0',
    'rail': '10',
    'goal': '11',
    'pillar': '12',
    'pole': '13',
    'floor': '14',
    'ground': '15',
    'grass': '16',
    'sand': '17',
    'athletic_field': '18',
    'road': '19',
    'path': '20',
    'crosswalk': '21',
    'building': '22',
    'house': '23',
    'bridge': '24',
    'tower': '25',
    'windmill': '26',
    'well_or_well_lid': '27',
    'other_construction': '28',
    'sky': '29',
    'mountain': '30',
    'stone': '31',
    'wood': '32',
    'ice': '33',
    'snowfield': '34',
    'grandstand': '35',
    'sea': '36',
    'river': '37',
    'lake': '38',
    'waterfall': '39',
    'water': '40',
    'billboard_or_Bulletin_Board': '41',
    'sculpture': '42',
    'pipeline': '43',
    'flag': '44',
    'parasol_or_umbrella': '45',
    'cushion_or_carpet': '46',
    'tent': '47',
    'roadblock': '48',
    'car': '49',
    'bus': '50',
    'truck': '51',
    'bicycle': '52',
    'motorcycle': '53',
    'wheeled_machine': '54',
    'ship_or_boat': '55',
    'raft': '56',
    'airplane': '57',
    'tyre': '58',
    'traffic_light': '59',
    'lamp': '60',
    'person': '61',
    'cat': '62',
    'dog': '63',
    'horse': '64',
    'cattle': '65',
    'other_animal': '66',
    'tree': '67',
    'flower': '68',
    'other_plant': '69',
    'toy': '70',
    'ball_net': '71',
    'backboard': '72',
    'skateboard': '73',
    'bat': '74',
    'ball': '75',
    'cupboard_or_showcase_or_storage_rack': '76',
    'box': '77',
    'traveling_case_or_trolley_case': '78',
    'basket': '79',
    'bag_or_package': '80',
    'trash_can': '81',
    'cage': '82',
    'plate': '83',
    'tub_or_bowl_or_pot': '84',
    'bottle_or_cup': '85',
    'barrel': '86',
    'fishbowl': '87',
    'bed': '88',
    'pillow': '89',
    'table_or_desk': '90',
    'chair_or_seat': '91',
    'bench': '92',
    'sofa': '93',
    'shelf': '94',
    'bathtub': '95',
    'gun': '96',
    'commode': '97',
    'roaster': '98',
    'other_machine': '99',
    'refrigerator': '100',
    'washing_machine': '101',
    'Microwave_oven': '102',
    'fan': '103',
    'curtain': '104',
    'textiles': '105',
    'clothes': '106',
    'painting_or_poster': '107',
    'mirror': '108',
    'flower_pot_or_vase': '109',
    'clock': '110',
    'book': '111',
    'tool': '112',
    'blackboard': '113',
    'tissue': '114',
    'screen_or_television': '115',
    'computer': '116',
    'printer': '117',
    'Mobile_phone': '118',
    'keyboard': '119',
    'other_electronic_product': '120',
    'fruit': '121',
    'food': '122',
    'instrument': '123',
    'train': '124'
}

vip_seg_label_to_id = {k: int(v) for k, v in vip_seg_label.items()}
vip_seg_id_to_label = {int(v): k for k, v in vip_seg_label.items()}

city_labels = [('road', 0, (128, 64, 128)), ('sidewalk', 1, (244, 35, 232)),
               ('building', 2, (70, 70, 70)), ('wall', 3, (102, 102, 156)),
               ('fence', 4, (190, 153, 153)), ('pole', 5, (153, 153, 153)),
               ('traffic light', 6, (250, 170, 30)),
               ('traffic sign', 7, (220, 220, 0)),
               ('vegetation', 8, (107, 142, 35)),
               ('terrain', 9, (152, 251, 152)), ('sky', 10, (70, 130, 180)),
               ('person', 11, (220, 20, 60)), ('rider', 12, (255, 0, 0)),
               ('car', 13, (0, 0, 142)), ('truck', 14, (0, 0, 70)),
               ('bus', 15, (0, 60, 100)), ('train', 16, (0, 80, 100)),
               ('motorcycle', 17, (0, 0, 230)), ('bicycle', 18, (119, 11, 32)),
               ('void', 19, (0, 0, 0)), ('void', 255, (0, 0, 0))]


def sha256num(num):
    hex = hashlib.sha256(str(num).encode('utf-8')).hexdigest()
    hex = hex[-6:]
    return int(hex, 16)


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def cityscapes_cat2rgb(cat_map):
    color_map = np.zeros_like(cat_map).astype(np.uint8)
    color_map = color_map[..., None].repeat(3, axis=-1)
    for each_class in city_labels:
        index = cat_map == each_class[1]
        if index.any():
            color_map[index] = each_class[2]
    return color_map


def trackmap2rgb(track_map):
    color_map = np.zeros_like(track_map).astype(np.uint8)
    color_map = color_map[..., None].repeat(3, axis=-1)
    for id_cur in np.unique(track_map):
        if id_cur == 0:
            continue
        color_map[track_map == id_cur] = id2rgb(sha256num(id_cur))
    return color_map


def draw_bbox_on_img(vis_img, bboxes):
    for index in range(bboxes.shape[0]):
        cv2.rectangle(
            vis_img, (int(bboxes[index][0]), int(bboxes[index][1])),
            (int(bboxes[index][2]), int(bboxes[index][3])), (0, 0, 255),
            thickness=1)
    return vis_img
