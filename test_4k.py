import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset


data_ms = MsDataset.load('DAMOXR/nerf_llff_data', subset_name='default', split='test')
data_path = data_ms.config_kwargs['split_config']['test']
scene = 'nerf_llff_data/fern'
datadir = os.path.join(data_path, scene)

data_dic = dict(
    datadir=datadir,
    dataset_type='llff',
    load_sr=1,
    factor=4,
    ndc=True,
    white_bkgd=False
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
    model='DAMOXR/cv_nerf-3d-reconstruction-4k-nerf_damo',
    data_type='llff'
    )

nerf_recon_4k(
    dict(data_cfg=data_dic, render_dir=render_dir))
### render results will be saved in render_dir
