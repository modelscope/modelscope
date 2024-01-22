lightglue_default_conf = {
    'features': 'superpoint',  # superpoint disk aliked sift
    'name': 'lightglue',  # just for interfacing
    'input_dim': 256,  # input descriptor dimension (autoselected from weights)
    'descriptor_dim': 256,
    'add_scale_ori': False,
    'n_layers': 9,
    'num_heads': 4,
    'flash': True,  # enable FlashAttention if available.
    'mp': False,  # enable mixed precision
    'depth_confidence': 0.95,  # early stopping, disable with -1
    'width_confidence': 0.99,  # point pruning, disable with -1
    'filter_threshold': 0.1,  # match threshold
    'weights': None,
}
