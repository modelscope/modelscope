import argparse
import os
import sys
import zipfile

from modelscope.hub.check_model import check_local_model_is_latest
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ThirdParty
from modelscope.utils.logger import get_logger

try:
    from tts_autolabel import AutoLabeling
except ImportError:
    raise ImportError('pls install tts-autolabel with \
                      "pip install tts-autolabel -f \
                      https://modelscope.oss-cn-beijing.aliyuncs.com/releases/repo.html"'
                      )

DEFAULT_RESOURCE_MODEL_ID = 'damo/speech_ptts_autolabel_16k'
logger = get_logger()


# Suggest params:
# --para_ids all --resource_revision v1.0.2 --input_wav data/test/audios/autolabel
# --work_dir ../ptts/test/diff2 --develop_mode 1 --stage 1 --process_num 2 --no_para --disable_enh
def run_auto_label(input_wav,
                   work_dir,
                   para_ids='all',
                   resource_model_id=DEFAULT_RESOURCE_MODEL_ID,
                   resource_revision=None,
                   gender='female',
                   stage=1,
                   process_num=4,
                   develop_mode=0,
                   has_para=False,
                   enable_enh=False):
    if not os.path.exists(input_wav):
        raise ValueError(f'input_wav: {input_wav} not exists')
    if not os.path.exists(work_dir):
        raise ValueError(f'work_dir: {work_dir} not exists')

    def _download_and_unzip_resource(model, model_revision=None):
        if os.path.exists(model):
            model_cache_dir = model if os.path.isdir(
                model) else os.path.dirname(model)
            check_local_model_is_latest(
                model_cache_dir,
                user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
        else:
            model_cache_dir = snapshot_download(
                model,
                revision=model_revision,
                user_agent={ThirdParty.KEY: 'speech_tts_autolabel'})
        if not os.path.exists(model_cache_dir):
            raise ValueError(f'model_cache_dir: {model_cache_dir} not exists')
        zip_file = os.path.join(model_cache_dir, 'model.zip')
        if not os.path.exists(zip_file):
            raise ValueError(f'zip_file: {zip_file} not exists')
        z = zipfile.ZipFile(zip_file)
        z.extractall(model_cache_dir)
        target_resource = os.path.join(model_cache_dir, 'model')
        return target_resource

    model_resource = _download_and_unzip_resource(resource_model_id,
                                                  resource_revision)
    auto_labeling = AutoLabeling(
        os.path.abspath(input_wav),
        model_resource,
        False,
        os.path.abspath(work_dir),
        gender,
        develop_mode,
        has_para,
        para_ids,
        stage,
        process_num,
        enable_enh=enable_enh)
    ret_code, report = auto_labeling.run()
    return ret_code, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--para_ids',
        default='all',
        help=
        'you can use this variable to config your auto labeling paragraph ids, \
        all means all in the dir, none means no paragraph 1 means 1 para only, \
        1 2 means 1 and 2, transcipt/prosody/wav should be named exactly the same!!!'
    )
    parser.add_argument(
        '--resource', type=str, default=DEFAULT_RESOURCE_MODEL_ID)
    parser.add_argument(
        '--resource_revision',
        type=str,
        default=None,
        help='resource directory')
    parser.add_argument('--input_wav', help='personal user input wav dir')
    parser.add_argument('--work_dir', help='autolabel work dir')
    parser.add_argument(
        '--gender', default='female', help='personal user gender')
    parser.add_argument('--develop_mode', type=int, default=1)
    parser.add_argument(
        '--stage',
        type=int,
        default=1,
        help='auto labeling stage, 0 means qualification and 1 means labeling')
    parser.add_argument(
        '--process_num',
        type=int,
        default=4,
        help='kaldi bin parallel execution process number')
    parser.add_argument(
        '--has_para', dest='has_para', action='store_true', help='paragraph')
    parser.add_argument(
        '--no_para',
        dest='has_para',
        action='store_false',
        help='no paragraph')
    parser.add_argument(
        '--enable_enh',
        dest='enable_enh',
        action='store_true',
        help='enable audio enhancement')
    parser.add_argument(
        '--disable_enh',
        dest='enable_enh',
        action='store_false',
        help='disable audio enhancement')
    parser.set_defaults(has_para=True)
    parser.set_defaults(enable_enh=False)
    args = parser.parse_args()
    logger.info(args.enable_enh)
    ret_code, report = run_auto_label(args.input_wav, args.work_dir,
                                      args.para_ids, args.resource,
                                      args.resource_revision, args.gender,
                                      args.stage, args.process_num,
                                      args.develop_mode, args.has_para,
                                      args.enable_enh)
    logger.info(f'ret_code={ret_code}')
    logger.info(f'report={report}')
