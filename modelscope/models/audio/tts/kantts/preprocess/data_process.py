# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import codecs
import os
import sys
import time

import yaml

from modelscope import __version__
from modelscope.models.audio.tts.kantts.datasets.dataset import (AmDataset,
                                                                 VocDataset)
from modelscope.utils.logger import get_logger
from .audio_processor.audio_processor import AudioProcessor
from .fp_processor import FpProcessor, is_fp_line
from .languages import languages
from .script_convertor.text_script_convertor import TextScriptConvertor

ROOT_PATH = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))  # NOQA: E402
sys.path.insert(0, os.path.dirname(ROOT_PATH))  # NOQA: E402

logging = get_logger()


def gen_metafile(
    voice_output_dir,
    fp_enable=False,
    badlist=None,
    split_ratio=0.98,
):

    voc_train_meta = os.path.join(voice_output_dir, 'train.lst')
    voc_valid_meta = os.path.join(voice_output_dir, 'valid.lst')
    if not os.path.exists(voc_train_meta) or not os.path.exists(
            voc_valid_meta):
        VocDataset.gen_metafile(
            os.path.join(voice_output_dir, 'wav'),
            voice_output_dir,
            split_ratio,
        )
        logging.info('Voc metafile generated.')

    raw_metafile = os.path.join(voice_output_dir, 'raw_metafile.txt')
    am_train_meta = os.path.join(voice_output_dir, 'am_train.lst')
    am_valid_meta = os.path.join(voice_output_dir, 'am_valid.lst')
    if not os.path.exists(am_train_meta) or not os.path.exists(am_valid_meta):
        AmDataset.gen_metafile(
            raw_metafile,
            voice_output_dir,
            am_train_meta,
            am_valid_meta,
            badlist,
            split_ratio,
        )
        logging.info('AM metafile generated.')

    if fp_enable:
        fpadd_metafile = os.path.join(voice_output_dir, 'fpadd_metafile.txt')
        am_train_meta = os.path.join(voice_output_dir, 'am_fpadd_train.lst')
        am_valid_meta = os.path.join(voice_output_dir, 'am_fpadd_valid.lst')
        if not os.path.exists(am_train_meta) or not os.path.exists(
                am_valid_meta):
            AmDataset.gen_metafile(
                fpadd_metafile,
                voice_output_dir,
                am_train_meta,
                am_valid_meta,
                badlist,
                split_ratio,
            )
            logging.info('AM fpaddmetafile generated.')

        fprm_metafile = os.path.join(voice_output_dir, 'fprm_metafile.txt')
        am_train_meta = os.path.join(voice_output_dir, 'am_fprm_train.lst')
        am_valid_meta = os.path.join(voice_output_dir, 'am_fprm_valid.lst')
        if not os.path.exists(am_train_meta) or not os.path.exists(
                am_valid_meta):
            AmDataset.gen_metafile(
                fprm_metafile,
                voice_output_dir,
                am_train_meta,
                am_valid_meta,
                badlist,
                split_ratio,
            )
            logging.info('AM fprmmetafile generated.')


def process_data(
    voice_input_dir,
    voice_output_dir,
    language_dir,
    audio_config,
    speaker_name=None,
    targetLang='PinYin',
    skip_script=False,
):
    foreignLang = 'EnUS'
    emo_tag_path = None

    phoneset_path = os.path.join(language_dir, targetLang,
                                 languages[targetLang]['phoneset_path'])
    posset_path = os.path.join(language_dir, targetLang,
                               languages[targetLang]['posset_path'])
    f2t_map_path = os.path.join(language_dir, targetLang,
                                languages[targetLang]['f2t_map_path'])
    s2p_map_path = os.path.join(language_dir, targetLang,
                                languages[targetLang]['s2p_map_path'])

    logging.info(f'phoneset_path={phoneset_path}')
    # dir of plain text/sentences for training byte based model
    plain_text_dir = os.path.join(voice_input_dir, 'text')

    if speaker_name is None:
        speaker_name = os.path.basename(voice_input_dir)

    if audio_config is not None:
        with open(audio_config, 'r') as f:
            config = yaml.load(f, Loader=yaml.Loader)

    config['create_time'] = time.strftime('%Y-%m-%d %H:%M:%S',
                                          time.localtime())
    config['modelscope_version'] = __version__

    with open(os.path.join(voice_output_dir, 'audio_config.yaml'), 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper, default_flow_style=None)

    if skip_script:
        logging.info('Skip script conversion')
    raw_metafile = None
    #  Script processor
    if not skip_script:
        if os.path.exists(plain_text_dir):
            TextScriptConvertor.turn_text_into_bytes(
                os.path.join(plain_text_dir, 'text.txt'),
                os.path.join(voice_output_dir, 'raw_metafile.txt'),
                speaker_name,
            )
            fp_enable = False
        else:
            tsc = TextScriptConvertor(
                phoneset_path,
                posset_path,
                targetLang,
                foreignLang,
                f2t_map_path,
                s2p_map_path,
                emo_tag_path,
                speaker_name,
            )
            tsc.process(
                os.path.join(voice_input_dir, 'prosody', 'prosody.txt'),
                os.path.join(voice_output_dir, 'Script.xml'),
                os.path.join(voice_output_dir, 'raw_metafile.txt'),
            )
            prosody = os.path.join(voice_input_dir, 'prosody', 'prosody.txt')
            # FP processor
            with codecs.open(prosody, 'r', 'utf-8') as f:
                lines = f.readlines()
                fp_enable = is_fp_line(lines[1])
        raw_metafile = os.path.join(voice_output_dir, 'raw_metafile.txt')

    if fp_enable:
        FP = FpProcessor()

        FP.process(
            voice_output_dir,
            prosody,
            raw_metafile,
        )
        logging.info('Processing fp done.')

    #  Audio processor
    ap = AudioProcessor(config['audio_config'])
    ap.process(
        voice_input_dir,
        voice_output_dir,
        raw_metafile,
    )

    logging.info('Processing done.')

    # Generate Voc&AM metafile
    gen_metafile(voice_output_dir, fp_enable, ap.badcase_list)
