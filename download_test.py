import shutil

from modelscope import dataset_snapshot_download, snapshot_download
from modelscope.utils.ms_tqdm import timing_decorator

# shutil.rmtree("/root/.cache/modelscope/datasets/AlexEz", ignore_errors=True)
shutil.rmtree('/root/.cache/modelscope/hub/AlexEz', ignore_errors=True)


@timing_decorator
def total_test():
    snapshot_download(model_id='AlexEz/test_model', max_workers=1)


total_test()
# dir = dataset_snapshot_download(dataset_id="AlexEz/image_dataset_example", max_workers=1)

# print(dir)

# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('clip-benchmark/wds_flickr8k', split='test')

# from huggingface_hub import snapshot_download
# snapshot_download(repo_id='gaia-benchmark/GAIA', repo_type='dataset', force_download=True)

# print(ds[0])
