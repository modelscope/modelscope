import os
from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


data_dic = dict(
    datadir='../datasets_nerf/nerf_llff_data/fern',
    dataset_type='llff',
    load_sr=1,
    llffhold=8,
    factor=4,
    ndc=True,
    white_bkgd=False,
    bd_factor=.75,
    width=None,
    height=None,
    spherify=False,
    movie_render_kwargs=dict(),
    load_depths=False
)

# data_dic = dict(
#     datadir='../datasets_nerf/nerf_synthetic/drums_4K',
#     dataset_type='blender',
#     load_sr=1,
#     factor=4,
#     half_res=4,
#     testskip=1,
#     ndc=False,
#     white_bkgd=True,
#     bd_factor=.75,
#     width=None,
#     height=None,
#     spherify=False,
#     movie_render_kwargs=dict(),
#     load_depths=False
# )

render_dir = 'exp'
### when use nerf-synthesis dataset, data_type should specify as 'blender'
nerf_recon_4k = pipeline(
    Tasks.nerf_recon_4k,
    model='./damo_4knerf',
    enc_ckpt_path="/home/admin/wzs/4K-NeRF/logs/llff/joint_fern_l1+gan/ckpt_saved/fine_100000.tar",
    dec_ckpt_path="/home/admin/wzs/4K-NeRF/logs/llff/joint_fern_l1+gan/ckpt_saved/sresrnet_100000.pth",
    # enc_ckpt_path="/home/admin/wzs/4K-NeRF/logs/syn/drums_4k/fine_180000.tar",
    # dec_ckpt_path="/home/admin/wzs/4K-NeRF/logs/syn/drums_4k/sresrnet_180000.pth",
    # model_dir='../4K-NeRF/logs/llff/joint_fern_l1+gan',
    data_type='llff'
    )

nerf_recon_4k(
    dict(data_cfg=data_dic, render_dir=render_dir))
### render results will be saved in render_dir
