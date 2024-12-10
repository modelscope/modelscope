# @Description: Options settings & configurations for GeoMVSNet.
# @Author: Zhe Zhang (doublez@stu.pku.edu.cn)
# @Affiliation: Peking University (PKU)
# @LastEditDate: 2023-09-07
# @https://github.com/doublez0108/geomvsnet

import argparse


def get_opts():
    parser = argparse.ArgumentParser(description='args')

    # global settings
    parser.add_argument(
        '--mode',
        default='train',
        help='train or test',
        choices=['train', 'test', 'val'])
    parser.add_argument(
        '--which_dataset',
        default='dtu',
        choices=['dtu', 'tnt', 'blendedmvs', 'general'],
        help='which dataset for using')

    parser.add_argument('--n_views', type=int, default=5, help='num of view')
    parser.add_argument('--levels', type=int, default=4, help='num of stages')
    parser.add_argument(
        '--hypo_plane_num_stages',
        type=str,
        default='8,8,4,4',
        help='num of hypothesis planes for each stage')
    parser.add_argument(
        '--depth_interal_ratio_stages',
        type=str,
        default='0.5,0.5,0.5,1',
        help='depth interals for each stage')
    parser.add_argument(
        '--feat_base_channel',
        type=int,
        default=8,
        help='channel num for base feature')
    parser.add_argument(
        '--reg_base_channel',
        type=int,
        default=8,
        help='channel num for regularization')
    parser.add_argument(
        '--group_cor_dim_stages',
        type=str,
        default='8,8,4,4',
        help='group correlation dim')

    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument(
        '--data_scale',
        type=str,
        choices=['mid', 'raw'],
        help='use mid or raw resolution')
    parser.add_argument('--trainpath', help='data path for training')
    parser.add_argument('--testpath', help='data path for testing')
    parser.add_argument('--trainlist', help='data list for training')
    parser.add_argument('--testlist', nargs='+', help='data list for testing')

    # training config
    parser.add_argument(
        '--stage_lw',
        type=str,
        default='1,1,1,1',
        help='loss weight for different stages')

    parser.add_argument(
        '--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='MS',
        help='scheduler for learning rate')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument(
        '--lrepochs',
        type=str,
        default='1,3,5,7,9,11,13,15:1.5',
        help='epoch ids to downscale lr and the downscale rate')
    parser.add_argument('--wd', type=float, default=0.0, help='weight decay')

    parser.add_argument(
        '--summary_freq',
        type=int,
        default=100,
        help='print and summary frequency')
    parser.add_argument(
        '--save_freq', type=int, default=1, help='save checkpoint frequency')
    parser.add_argument(
        '--eval_freq', type=int, default=1, help='eval frequency')

    parser.add_argument(
        '--robust_train', action='store_true', help='robust training')

    # testing config
    parser.add_argument(
        '--split',
        type=str,
        choices=['intermediate', 'advanced'],
        help='intermediate|advanced for tanksandtemples')
    parser.add_argument(
        '--img_mode',
        type=str,
        default='resize',
        choices=['resize', 'crop'],
        help='image resolution matching strategy for TNT dataset')
    parser.add_argument(
        '--cam_mode',
        type=str,
        default='origin',
        choices=['origin', 'short_range'],
        help='camera parameter strategy for TNT dataset')

    parser.add_argument(
        '--loadckpt', default=None, help='load a specific checkpoint')
    parser.add_argument(
        '--logdir',
        default='./checkpoints/debug',
        help='the directory to save checkpoints/logs')
    parser.add_argument(
        '--nolog', action='store_true', help='do not log into .log file')
    parser.add_argument(
        '--notensorboard',
        action='store_true',
        help='do not log into tensorboard')
    parser.add_argument(
        '--save_conf_all_stages',
        action='store_true',
        help='save confidence maps for all stages')
    parser.add_argument('--outdir', default='./outputs', help='output dir')
    parser.add_argument(
        '--resume', action='store_true', help='continue to train the model')

    # pytorch config
    parser.add_argument('--device', default='cuda', help='device to use')
    parser.add_argument(
        '--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument(
        '--pin_m', action='store_true', help='data loader pin memory')
    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args()
