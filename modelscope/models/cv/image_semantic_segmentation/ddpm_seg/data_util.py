# The implementation here is adopted from ddpm-segmentation,
# originally Apache 2.0 License and publicly available at https://github.com/yandex-research/ddpm-segmentation


def get_palette(category):
    if category == 'ffhq_34':
        return ffhq_palette
    elif category == 'bedroom_28':
        return bedroom_palette
    elif category == 'cat_15':
        return cat_palette
    elif category == 'horse_21':
        return horse_palette
    elif category == 'ade_bedroom_30':
        return ade_bedroom_30_palette
    elif category == 'celeba_19':
        return celeba_palette


def get_class_names(category):
    if category == 'ffhq_34':
        return ffhq_class
    elif category == 'bedroom_28':
        return bedroom_class
    elif category == 'cat_15':
        return cat_class
    elif category == 'horse_21':
        return horse_class
    elif category == 'ade_bedroom_30':
        return ade_bedroom_30_class
    elif category == 'celeba_19':
        return celeba_class


###############
# Class names #
###############

bedroom_class = [
    'background', 'bed', 'bed***footboard', 'bed***headboard',
    'bed***side rail', 'carpet', 'ceiling', 'chandelier / ceiling fan blade',
    'curtain', 'cushion', 'floor', 'table/nightstand/dresser',
    'table/nightstand/dresser***top', 'picture / mirrow', 'pillow',
    'lamp***column', 'lamp***shade', 'wall', 'window', 'curtain rod',
    'window***frame', 'chair', 'picture / mirror***frame', 'plinth',
    'door / door frame', 'pouf', 'wardrobe', 'plant', 'table staff'
]

ffhq_class = [
    'background', 'head', 'head***cheek', 'head***chin', 'head***ear',
    'head***ear***helix', 'head***ear***lobule', 'head***eye***bottom lid',
    'head***eye***eyelashes', 'head***eye***iris', 'head***eye***pupil',
    'head***eye***sclera', 'head***eye***tear duct', 'head***eye***top lid',
    'head***eyebrow', 'head***forehead', 'head***frown', 'head***hair',
    'head***hair***sideburns', 'head***jaw', 'head***moustache',
    'head***mouth***inferior lip', 'head***mouth***oral commissure',
    'head***mouth***superior lip', 'head***mouth***teeth', 'head***neck',
    'head***nose', 'head***nose***ala of nose', 'head***nose***bridge',
    'head***nose***nose tip', 'head***nose***nostril', 'head***philtrum',
    'head***temple', 'head***wrinkles'
]

cat_class = [
    'background', 'back', 'belly', 'chest', 'leg', 'paw', 'head', 'ear', 'eye',
    'mouth', 'tongue', 'nose', 'tail', 'whiskers', 'neck'
]

horse_class = [
    'background', 'person', 'back', 'barrel', 'bridle', 'chest', 'ear', 'eye',
    'forelock', 'head', 'hoof', 'leg', 'mane', 'muzzle', 'neck', 'nostril',
    'tail', 'thigh', 'saddle', 'shoulder', 'leg protection'
]

celeba_class = [
    'background', 'cloth', 'ear_r', 'eye_g', 'hair', 'hat', 'l_brow', 'l_ear',
    'l_eye', 'l_lip', 'mouth', 'neck', 'neck_l', 'nose', 'r_brow', 'r_ear',
    'r_eye', 'skin', 'u_lip'
]

ade_bedroom_50_class = [
    'wall', 'bed', 'floor', 'table', 'lamp', 'ceiling', 'painting',
    'windowpane', 'pillow', 'curtain', 'cushion', 'door', 'chair', 'cabinet',
    'chest', 'mirror', 'rug', 'armchair', 'book', 'sconce', 'plant',
    'wardrobe', 'clock', 'light', 'flower', 'vase', 'fan', 'box', 'shelf',
    'television', 'blind', 'pot', 'ottoman', 'sofa', 'desk', 'basket',
    'blanket', 'coffee', 'plaything', 'radiator', 'tray', 'stool', 'bottle',
    'chandelier', 'fireplacel', 'towel', 'railing', 'canopy', 'glass', 'plate'
]

ade_bedroom_40_class = ade_bedroom_50_class[:40]
ade_bedroom_30_class = ade_bedroom_50_class[:30]

###########
# Palette #
###########

ffhq_palette = [
    1.0000, 1.0000, 1.0000, 0.4420, 0.5100, 0.4234, 0.8562, 0.9537, 0.3188,
    0.2405, 0.4699, 0.9918, 0.8434, 0.9329, 0.7544, 0.3748, 0.7917, 0.3256,
    0.0190, 0.4943, 0.3782, 0.7461, 0.0137, 0.5684, 0.1644, 0.2402, 0.7324,
    0.0200, 0.4379, 0.4100, 0.5853, 0.8880, 0.6137, 0.7991, 0.9132, 0.9720,
    0.6816, 0.6237, 0.8562, 0.9981, 0.4692, 0.3849, 0.5351, 0.8242, 0.2731,
    0.1747, 0.3626, 0.8345, 0.5323, 0.6668, 0.4922, 0.2122, 0.3483, 0.4707,
    0.6844, 0.1238, 0.1452, 0.3882, 0.4664, 0.1003, 0.2296, 0.0401, 0.3030,
    0.5751, 0.5467, 0.9835, 0.1308, 0.9628, 0.0777, 0.2849, 0.1846, 0.2625,
    0.9764, 0.9420, 0.6628, 0.3893, 0.4456, 0.6433, 0.8705, 0.3957, 0.0963,
    0.6117, 0.9702, 0.0247, 0.3668, 0.6694, 0.3117, 0.6451, 0.7302, 0.9542,
    0.6171, 0.1097, 0.9053, 0.3377, 0.4950, 0.7284, 0.1655, 0.9254, 0.6557,
    0.9450, 0.6721, 0.6162
]

ffhq_palette = [int(item * 255) for item in ffhq_palette]

bedroom_palette = [
    255,
    255,
    255,  # bg
    238,
    229,
    102,  # bed
    255,
    72,
    69,  # bed footboard
    124,
    99,
    34,  # bed headboard
    193,
    127,
    15,  # bed side rail
    106,
    177,
    21,  # carpet
    248,
    213,
    43,  # ceiling
    252,
    155,
    83,  # chandelier / ceiling fan blade
    220,
    147,
    77,  # curtain
    99,
    83,
    3,  # cushion
    116,
    116,
    138,  # floor
    63,
    182,
    24,  # table/nightstand/dresser
    200,
    226,
    37,  # table/nightstand/dresser top
    225,
    184,
    161,  # picture / mirrow
    233,
    5,
    219,  # pillow
    142,
    172,
    248,  # lamp column
    153,
    112,
    146,  # lamp shade
    38,
    112,
    254,  # wall
    229,
    30,
    141,  # window
    99,
    205,
    255,  # curtain rod
    74,
    59,
    83,  # window frame
    186,
    9,
    0,  # chair
    107,
    121,
    0,  # picture / mirrow frame
    0,
    194,
    160,  # plinth
    255,
    170,
    146,  # door / door frame
    255,
    144,
    201,  # pouf
    185,
    3,
    170,  # wardrobe
    221,
    239,
    255,  # plant
    0,
    0,
    53,  # table staff
]

cat_palette = [
    255, 255, 255, 190, 153, 153, 250, 170, 30, 220, 220, 0, 107, 142, 35, 102,
    102, 156, 152, 251, 152, 119, 11, 32, 244, 35, 232, 220, 20, 60, 52, 83,
    84, 194, 87, 125, 143, 176, 255, 31, 102, 211, 104, 131, 101
]

horse_palette = [
    255, 255, 255, 255, 74, 70, 0, 137, 65, 0, 111, 166, 163, 0, 89, 255, 219,
    229, 122, 73, 0, 0, 0, 166, 99, 255, 172, 183, 151, 98, 0, 77, 67, 143,
    176, 255, 241, 38, 110, 27, 210, 105, 128, 150, 147, 228, 230, 158, 160,
    136, 106, 79, 198, 1, 59, 93, 255, 115, 214, 209, 255, 47, 128
]

celeba_palette = [
    255,
    255,
    255,  # 0 background
    238,
    229,
    102,  # 1 cloth
    250,
    150,
    50,  # 2 ear_r
    124,
    99,
    34,  # 3 eye_g
    193,
    127,
    15,  # 4 hair
    225,
    96,
    18,  # 5 hat
    220,
    147,
    77,  # 6 l_brow
    99,
    83,
    3,  # 7 l_ear
    116,
    116,
    138,  # 8 l_eye
    200,
    226,
    37,  # 9 l_lip
    225,
    184,
    161,  # 10 mouth
    142,
    172,
    248,  # 11 neck
    153,
    112,
    146,  # 12 neck_l
    38,
    112,
    254,  # 13 nose
    229,
    30,
    141,  # 14 r_brow
    52,
    83,
    84,  # 15 r_ear
    194,
    87,
    125,  # 16 r_eye
    248,
    213,
    42,  # 17 skin
    31,
    102,
    211,  # 18 u_lip
]

ade_bedroom_50_palette = [
    240, 156, 206, 69, 88, 93, 240, 49, 184, 27, 107, 126, 50, 82, 241, 54,
    250, 147, 156, 213, 3, 176, 108, 79, 251, 150, 149, 66, 51, 34, 210, 97,
    53, 30, 53, 102, 232, 164, 118, 204, 150, 17, 101, 86, 178, 249, 20, 213,
    54, 35, 82, 157, 68, 216, 58, 161, 73, 174, 67, 67, 193, 181, 78, 169, 60,
    178, 220, 204, 166, 4, 127, 85, 245, 106, 216, 222, 172, 168, 84, 148, 105,
    137, 220, 89, 68, 252, 126, 29, 193, 187, 74, 40, 101, 52, 71, 61, 38, 92,
    205, 40, 104, 224, 146, 74, 160, 69, 43, 220, 70, 78, 213, 249, 93, 254,
    235, 71, 119, 193, 255, 102, 152, 55, 238, 133, 12, 223, 106, 116, 123, 86,
    14, 174, 244, 160, 161, 142, 105, 60, 153, 61, 124, 195, 156, 253, 241, 84,
    222, 202, 171, 227
]

ade_bedroom_40_palette = ade_bedroom_50_palette[:120]
ade_bedroom_30_palette = ade_bedroom_50_palette[:90]
