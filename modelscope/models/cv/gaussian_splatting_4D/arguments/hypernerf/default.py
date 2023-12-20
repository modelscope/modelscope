ModelHiddenParams = dict(
    kplanes_config = {
     'grid_dimensions': 2,
     'input_coordinate_dim': 4,
     'output_coordinate_dim': 16,
     'resolution': [64, 64, 64, 150]
    },
    multires = [1,2,4,8],
    defor_depth = 2,
    net_width = 256,
    plane_tv_weight = 0.0002,
    time_smoothness_weight = 0.001,
    l1_time_planes =  0.001,

)
OptimizationParams = dict(
    dataloader=False,
    iterations = 60_000,
    coarse_iterations = 3000,
    densify_until_iter = 45_000,
    opacity_reset_interval = 6000,
    # position_lr_init = 0.00016,
    # position_lr_final = 0.0000016,
    # position_lr_delay_mult = 0.01,
    # position_lr_max_steps = 60_000,
    deformation_lr_init = 0.0016,
    deformation_lr_final = 0.00016,
    deformation_lr_delay_mult = 0.01,
    grid_lr_init = 0.016,
    grid_lr_final = 0.0016,
    # densify_until_iter = 50_000,
    opacity_threshold_coarse = 0.005,
    opacity_threshold_fine_init = 0.005,
    opacity_threshold_fine_after = 0.005,
    # pruning_interval = 2000
)