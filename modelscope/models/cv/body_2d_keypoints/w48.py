# The implementation is based on HRNET, available at https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation.

cfg_128x128_15 = {
    'DATASET': {
        'TYPE': 'DAMO',
        'PARENT_IDS': [0, 0, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, 1],
        'LEFT_IDS': [2, 3, 4, 8, 9, 10],
        'RIGHT_IDS': [5, 6, 7, 11, 12, 13],
        'SPINE_IDS': [0, 1, 14]
    },
    'MODEL': {
        'INIT_WEIGHTS': True,
        'NAME': 'pose_hrnet',
        'NUM_JOINTS': 15,
        'PRETRAINED': '',
        'TARGET_TYPE': 'gaussian',
        'IMAGE_SIZE': [128, 128],
        'HEATMAP_SIZE': [32, 32],
        'SIGMA': 2.0,
        'EXTRA': {
            'PRETRAINED_LAYERS': [
                'conv1', 'bn1', 'conv2', 'bn2', 'layer1', 'transition1',
                'stage2', 'transition2', 'stage3', 'transition3', 'stage4'
            ],
            'FINAL_CONV_KERNEL':
            1,
            'STAGE2': {
                'NUM_MODULES': 1,
                'NUM_BRANCHES': 2,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4],
                'NUM_CHANNELS': [48, 96],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE3': {
                'NUM_MODULES': 4,
                'NUM_BRANCHES': 3,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192],
                'FUSE_METHOD': 'SUM'
            },
            'STAGE4': {
                'NUM_MODULES': 3,
                'NUM_BRANCHES': 4,
                'BLOCK': 'BASIC',
                'NUM_BLOCKS': [4, 4, 4, 4],
                'NUM_CHANNELS': [48, 96, 192, 384],
                'FUSE_METHOD': 'SUM'
            },
        }
    }
}
