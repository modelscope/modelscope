# Copyright (c) Alibaba, Inc. and its affiliates.

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np
import yaml
from tqdm import tqdm

from modelscope.utils.logger import get_logger
from .core.dsp import (load_wav, melspectrogram, save_wav, trim_silence,
                       trim_silence_with_interval)
from .core.utils import (align_length, average_by_duration, compute_mean,
                         compute_std, encode_16bits, f0_norm_mean_std,
                         get_energy, get_pitch, norm_mean_std,
                         parse_interval_file, volume_normalize)

logging = get_logger()

default_audio_config = {
    #  Preprocess
    'wav_normalize': True,
    'trim_silence': True,
    'trim_silence_threshold_db': 60,
    'preemphasize': False,
    #  Feature extraction
    'sampling_rate': 24000,
    'hop_length': 240,
    'win_length': 1024,
    'n_mels': 80,
    'n_fft': 1024,
    'fmin': 50.0,
    'fmax': 7600.0,
    'min_level_db': -100,
    'ref_level_db': 20,
    'phone_level_feature': True,
    'num_workers': 16,
    #  Normalization
    'norm_type': 'mean_std',  # 'mean_std', 'global norm'
    'max_norm': 1.0,
    'symmetric': False,
}


class AudioProcessor:

    def __init__(self, config=None):
        if not isinstance(config, dict):
            logging.warning(
                '[AudioProcessor] config is not a dict, fall into default config.'
            )
            self.config = default_audio_config
        else:
            self.config = config

        for key in self.config:
            setattr(self, key, self.config[key])

        self.min_wav_length = int(self.config['sampling_rate'] * 0.5)

        self.badcase_list = []
        self.pcm_dict = {}
        self.mel_dict = {}
        self.f0_dict = {}
        self.uv_dict = {}
        self.nccf_dict = {}
        self.f0uv_dict = {}
        self.energy_dict = {}
        self.dur_dict = {}
        logging.info('[AudioProcessor] Initialize AudioProcessor.')
        logging.info('[AudioProcessor] config params:')
        for key in self.config:
            logging.info('[AudioProcessor] %s: %s', key, self.config[key])

    def calibrate_SyllableDuration(self, raw_dur_dir, raw_metafile,
                                   out_cali_duration_dir):
        with open(raw_metafile, 'r') as f:
            lines = f.readlines()

        output_dur_dir = out_cali_duration_dir
        os.makedirs(output_dur_dir, exist_ok=True)

        for line in lines:
            line = line.strip()
            index, symbols = line.split('\t')
            symbols = [
                symbol.strip('{').strip('}').split('$')[0]
                for symbol in symbols.strip().split(' ')
            ]
            dur_file = os.path.join(raw_dur_dir, index + '.npy')
            phone_file = os.path.join(raw_dur_dir, index + '.phone')
            if not os.path.exists(dur_file) or not os.path.exists(phone_file):
                logging.warning(
                    '[AudioProcessor] dur file or phone file not exists: %s',
                    index)
                continue
            with open(phone_file, 'r') as f:
                phones = f.readlines()
            dur = np.load(dur_file)
            cali_duration = []

            dur_idx = 0
            syll_idx = 0

            while dur_idx < len(dur) and syll_idx < len(symbols):
                if phones[dur_idx].strip() == 'sil':
                    dur_idx += 1
                    continue

                if phones[dur_idx].strip(
                ) == 'sp' and symbols[syll_idx][0] != '#':
                    dur_idx += 1
                    continue

                if symbols[syll_idx] in ['ga', 'go', 'ge']:
                    cali_duration.append(0)
                    syll_idx += 1
                    #  print("NONE", symbols[syll_idx], 0)
                    continue

                if symbols[syll_idx][0] == '#':
                    if phones[dur_idx].strip() != 'sp':
                        cali_duration.append(0)
                        #  print("NONE", symbols[syll_idx], 0)
                        syll_idx += 1
                        continue
                    else:
                        cali_duration.append(dur[dur_idx])
                        #  print(phones[dur_idx].strip(), symbols[syll_idx], dur[dur_idx])
                        dur_idx += 1
                        syll_idx += 1
                        continue
                # A corresponding phone is found
                cali_duration.append(dur[dur_idx])
                #  print(phones[dur_idx].strip(), symbols[syll_idx], dur[dur_idx])
                dur_idx += 1
                syll_idx += 1
            # Add #4 phone duration
            cali_duration.append(0)
            if len(cali_duration) != len(symbols):
                logging.error('[Duration Calibrating] Syllable duration {}\
                        is not equal to the number of symbols {}, index: {}'.
                              format(len(cali_duration), len(symbols), index))
                continue

            #  Align with mel frames
            durs = np.array(cali_duration)
            if len(self.mel_dict) > 0:
                pair_mel = self.mel_dict.get(index, None)
                if pair_mel is None:
                    logging.warning(
                        '[AudioProcessor] Interval file %s  has no corresponding mel',
                        index,
                    )
                    continue
                mel_frames = pair_mel.shape[0]
                dur_frames = np.sum(durs)
                if np.sum(durs) > mel_frames:
                    durs[-2] -= dur_frames - mel_frames
                elif np.sum(durs) < mel_frames:
                    durs[-2] += mel_frames - np.sum(durs)

                if durs[-2] < 0:
                    logging.error(
                        '[AudioProcessor] Duration calibrating failed for %s, mismatch frames %s',
                        index,
                        durs[-2],
                    )
                    self.badcase_list.append(index)
                    continue

            self.dur_dict[index] = durs

            np.save(
                os.path.join(output_dur_dir, index + '.npy'),
                self.dur_dict[index])

    def amp_normalize(self, src_wav_dir, out_wav_dir):
        if self.wav_normalize:
            logging.info('[AudioProcessor] Amplitude normalization started')
            os.makedirs(out_wav_dir, exist_ok=True)
            res = volume_normalize(src_wav_dir, out_wav_dir)
            logging.info('[AudioProcessor] Amplitude normalization finished')
            return res
        else:
            logging.info('[AudioProcessor] No amplitude normalization')
            os.symlink(src_wav_dir, out_wav_dir, target_is_directory=True)
            return True

    def get_pcm_dict(self, src_wav_dir):
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        if len(self.pcm_dict) > 0:
            return self.pcm_dict

        logging.info('[AudioProcessor] Start to load pcm from %s', src_wav_dir)
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_path in wav_list:
                future = executor.submit(load_wav, wav_path,
                                         self.sampling_rate)
                future.add_done_callback(lambda p: progress.update())
                wav_name = os.path.splitext(os.path.basename(wav_path))[0]
                futures.append((future, wav_name))
            for future, wav_name in futures:
                pcm = future.result()
                if len(pcm) < self.min_wav_length:
                    logging.warning('[AudioProcessor] %s is too short, skip',
                                    wav_name)
                    self.badcase_list.append(wav_name)
                    continue
                self.pcm_dict[wav_name] = pcm

        return self.pcm_dict

    def trim_silence_wav(self, src_wav_dir, out_wav_dir=None):
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        logging.info('[AudioProcessor] Trim silence started')
        if out_wav_dir is None:
            out_wav_dir = src_wav_dir
        else:
            os.makedirs(out_wav_dir, exist_ok=True)
        pcm_dict = self.get_pcm_dict(src_wav_dir)
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_basename, pcm_data in pcm_dict.items():
                future = executor.submit(
                    trim_silence,
                    pcm_data,
                    self.trim_silence_threshold_db,
                    self.hop_length,
                    self.win_length,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append((future, wav_basename))
        for future, wav_basename in tqdm(futures):
            pcm = future.result()
            if len(pcm) < self.min_wav_length:
                logging.warning('[AudioProcessor] %s is too short, skip',
                                wav_basename)
                self.badcase_list.append(wav_basename)
                self.pcm_dict.pop(wav_basename)
                continue
            self.pcm_dict[wav_basename] = pcm
            save_wav(
                self.pcm_dict[wav_basename],
                os.path.join(out_wav_dir, wav_basename + '.wav'),
                self.sampling_rate,
            )

        logging.info('[AudioProcessor] Trim silence finished')
        return True

    def trim_silence_wav_with_interval(self,
                                       src_wav_dir,
                                       dur_dir,
                                       out_wav_dir=None):
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        logging.info('[AudioProcessor] Trim silence with interval started')
        if out_wav_dir is None:
            out_wav_dir = src_wav_dir
        else:
            os.makedirs(out_wav_dir, exist_ok=True)
        pcm_dict = self.get_pcm_dict(src_wav_dir)
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_basename, pcm_data in pcm_dict.items():
                future = executor.submit(
                    trim_silence_with_interval,
                    pcm_data,
                    self.dur_dict.get(wav_basename, None),
                    self.hop_length,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append((future, wav_basename))
        for future, wav_basename in tqdm(futures):
            trimed_pcm = future.result()
            if trimed_pcm is None:
                continue
            if len(trimed_pcm) < self.min_wav_length:
                logging.warning('[AudioProcessor] %s is too short, skip',
                                wav_basename)
                self.badcase_list.append(wav_basename)
                self.pcm_dict.pop(wav_basename)
                continue
            self.pcm_dict[wav_basename] = trimed_pcm
            save_wav(
                self.pcm_dict[wav_basename],
                os.path.join(out_wav_dir, wav_basename + '.wav'),
                self.sampling_rate,
            )

        logging.info('[AudioProcessor] Trim silence finished')
        return True

    def mel_extract(self, src_wav_dir, out_feature_dir):
        os.makedirs(out_feature_dir, exist_ok=True)
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        pcm_dict = self.get_pcm_dict(src_wav_dir)

        logging.info('[AudioProcessor] Melspec extraction started')

        # Get global normed mel spec
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_basename, pcm_data in pcm_dict.items():
                future = executor.submit(
                    melspectrogram,
                    pcm_data,
                    self.sampling_rate,
                    self.n_fft,
                    self.hop_length,
                    self.win_length,
                    self.n_mels,
                    self.max_norm,
                    self.min_level_db,
                    self.ref_level_db,
                    self.fmin,
                    self.fmax,
                    self.symmetric,
                    self.preemphasize,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append((future, wav_basename))

            for future, wav_basename in futures:
                result = future.result()
                if result is None:
                    logging.warning(
                        '[AudioProcessor] Melspec extraction failed for %s',
                        wav_basename,
                    )
                    self.badcase_list.append(wav_basename)
                else:
                    melspec = result
                    self.mel_dict[wav_basename] = melspec

        logging.info('[AudioProcessor] Melspec extraction finished')

        #  FIXME: is this step necessary?
        #  Do mean std norm on global-normed melspec
        logging.info('Melspec statistic proceeding...')
        mel_mean = compute_mean(list(self.mel_dict.values()), dims=self.n_mels)
        mel_std = compute_std(
            list(self.mel_dict.values()), mel_mean, dims=self.n_mels)
        logging.info('Melspec statistic done')
        np.savetxt(
            os.path.join(out_feature_dir, 'mel_mean.txt'),
            mel_mean,
            fmt='%.6f')
        np.savetxt(
            os.path.join(out_feature_dir, 'mel_std.txt'), mel_std, fmt='%.6f')
        logging.info(
            '[AudioProcessor] melspec mean and std saved to:\n{},\n{}'.format(
                os.path.join(out_feature_dir, 'mel_mean.txt'),
                os.path.join(out_feature_dir, 'mel_std.txt'),
            ))

        logging.info('[AudioProcessor] Melspec mean std norm is proceeding...')
        for wav_basename in self.mel_dict:
            melspec = self.mel_dict[wav_basename]
            norm_melspec = norm_mean_std(melspec, mel_mean, mel_std)
            np.save(
                os.path.join(out_feature_dir, wav_basename + '.npy'),
                norm_melspec)

        logging.info('[AudioProcessor] Melspec normalization finished')
        logging.info('[AudioProcessor] Normed Melspec saved to %s',
                     out_feature_dir)

        return True

    def duration_generate(self, src_interval_dir, out_feature_dir):
        os.makedirs(out_feature_dir, exist_ok=True)
        interval_list = glob(os.path.join(src_interval_dir, '*.interval'))

        logging.info('[AudioProcessor] Duration generation started')
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(interval_list)) as progress:
            futures = []
            for interval_file_path in interval_list:
                future = executor.submit(
                    parse_interval_file,
                    interval_file_path,
                    self.sampling_rate,
                    self.hop_length,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append((future,
                                os.path.splitext(
                                    os.path.basename(interval_file_path))[0]))

            logging.info(
                '[AudioProcessor] Duration align with mel is proceeding...')
            for future, wav_basename in futures:
                result = future.result()
                if result is None:
                    logging.warning(
                        '[AudioProcessor] Duration generate failed for %s',
                        wav_basename)
                    self.badcase_list.append(wav_basename)
                else:
                    durs, phone_list = result
                    #  Algin length with melspec
                    if len(self.mel_dict) > 0:
                        pair_mel = self.mel_dict.get(wav_basename, None)
                        if pair_mel is None:
                            logging.warning(
                                '[AudioProcessor] Interval file %s  has no corresponding mel',
                                wav_basename,
                            )
                            continue
                        mel_frames = pair_mel.shape[0]
                        dur_frames = np.sum(durs)
                        if np.sum(durs) > mel_frames:
                            durs[-1] -= dur_frames - mel_frames
                        elif np.sum(durs) < mel_frames:
                            durs[-1] += mel_frames - np.sum(durs)

                        if durs[-1] < 0:
                            logging.error(
                                '[AudioProcessor] Duration align failed for %s, mismatch frames %s',
                                wav_basename,
                                durs[-1],
                            )
                            self.badcase_list.append(wav_basename)
                            continue

                    self.dur_dict[wav_basename] = durs

                    np.save(
                        os.path.join(out_feature_dir, wav_basename + '.npy'),
                        durs)
                    with open(
                            os.path.join(out_feature_dir,
                                         wav_basename + '.phone'), 'w') as f:
                        f.write('\n'.join(phone_list))
        logging.info('[AudioProcessor] Duration generate finished')

        return True

    def pitch_extract(self, src_wav_dir, out_f0_dir, out_frame_f0_dir,
                      out_frame_uv_dir):
        os.makedirs(out_f0_dir, exist_ok=True)
        os.makedirs(out_frame_f0_dir, exist_ok=True)
        os.makedirs(out_frame_uv_dir, exist_ok=True)
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        pcm_dict = self.get_pcm_dict(src_wav_dir)
        mel_dict = self.mel_dict

        logging.info('[AudioProcessor] Pitch extraction started')
        # Get raw pitch
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_basename, pcm_data in pcm_dict.items():
                future = executor.submit(
                    get_pitch,
                    encode_16bits(pcm_data),
                    self.sampling_rate,
                    self.hop_length,
                )
                future.add_done_callback(lambda p: progress.update())
                futures.append((future, wav_basename))

            logging.info(
                '[AudioProcessor] Pitch align with mel is proceeding...')
            for future, wav_basename in futures:
                result = future.result()
                if result is None:
                    logging.warning(
                        '[AudioProcessor] Pitch extraction failed for %s',
                        wav_basename)
                    self.badcase_list.append(wav_basename)
                else:
                    f0, uv, f0uv = result
                    if len(mel_dict) > 0:
                        f0 = align_length(f0, mel_dict.get(wav_basename, None))
                        uv = align_length(uv, mel_dict.get(wav_basename, None))
                        f0uv = align_length(f0uv,
                                            mel_dict.get(wav_basename, None))

                    if f0 is None or uv is None or f0uv is None:
                        logging.warning(
                            '[AudioProcessor] Pitch length mismatch with mel in %s',
                            wav_basename,
                        )
                        self.badcase_list.append(wav_basename)
                        continue
                    self.f0_dict[wav_basename] = f0
                    self.uv_dict[wav_basename] = uv
                    self.f0uv_dict[wav_basename] = f0uv

        #  Normalize f0
        logging.info('[AudioProcessor] Pitch normalization is proceeding...')
        f0_mean = compute_mean(list(self.f0uv_dict.values()), dims=1)
        f0_std = compute_std(list(self.f0uv_dict.values()), f0_mean, dims=1)
        np.savetxt(
            os.path.join(out_f0_dir, 'f0_mean.txt'), f0_mean, fmt='%.6f')
        np.savetxt(os.path.join(out_f0_dir, 'f0_std.txt'), f0_std, fmt='%.6f')
        logging.info(
            '[AudioProcessor] f0 mean and std saved to:\n{},\n{}'.format(
                os.path.join(out_f0_dir, 'f0_mean.txt'),
                os.path.join(out_f0_dir, 'f0_std.txt'),
            ))
        logging.info('[AudioProcessor] Pitch mean std norm is proceeding...')
        for wav_basename in self.f0uv_dict:
            f0 = self.f0uv_dict[wav_basename]
            norm_f0 = f0_norm_mean_std(f0, f0_mean, f0_std)
            self.f0uv_dict[wav_basename] = norm_f0

        for wav_basename in self.f0_dict:
            f0 = self.f0_dict[wav_basename]
            norm_f0 = f0_norm_mean_std(f0, f0_mean, f0_std)
            self.f0_dict[wav_basename] = norm_f0

        #  save frame f0 to a specific dir
        for wav_basename in self.f0_dict:
            np.save(
                os.path.join(out_frame_f0_dir, wav_basename + '.npy'),
                self.f0_dict[wav_basename].reshape(-1),
            )

        for wav_basename in self.uv_dict:
            np.save(
                os.path.join(out_frame_uv_dir, wav_basename + '.npy'),
                self.uv_dict[wav_basename].reshape(-1),
            )

        #  phone level average
        #  if there is no duration then save the frame-level f0
        if self.phone_level_feature and len(self.dur_dict) > 0:
            logging.info(
                '[AudioProcessor] Pitch turn to phone-level is proceeding...')
            with ProcessPoolExecutor(
                    max_workers=self.num_workers) as executor, tqdm(
                        total=len(self.f0uv_dict)) as progress:
                futures = []
                for wav_basename in self.f0uv_dict:
                    future = executor.submit(
                        average_by_duration,
                        self.f0uv_dict.get(wav_basename, None),
                        self.dur_dict.get(wav_basename, None),
                    )
                    future.add_done_callback(lambda p: progress.update())
                    futures.append((future, wav_basename))

                for future, wav_basename in futures:
                    result = future.result()
                    if result is None:
                        logging.warning(
                            '[AudioProcessor] Pitch extraction failed in phone level avg for: %s',
                            wav_basename,
                        )
                        self.badcase_list.append(wav_basename)
                    else:
                        avg_f0 = result
                        self.f0uv_dict[wav_basename] = avg_f0

        for wav_basename in self.f0uv_dict:
            np.save(
                os.path.join(out_f0_dir, wav_basename + '.npy'),
                self.f0uv_dict[wav_basename].reshape(-1),
            )

        logging.info('[AudioProcessor] Pitch normalization finished')
        logging.info('[AudioProcessor] Normed f0 saved to %s', out_f0_dir)
        logging.info('[AudioProcessor] Pitch extraction finished')

        return True

    def energy_extract(self, src_wav_dir, out_energy_dir,
                       out_frame_energy_dir):
        os.makedirs(out_energy_dir, exist_ok=True)
        os.makedirs(out_frame_energy_dir, exist_ok=True)
        wav_list = glob(os.path.join(src_wav_dir, '*.wav'))
        pcm_dict = self.get_pcm_dict(src_wav_dir)
        mel_dict = self.mel_dict

        logging.info('[AudioProcessor] Energy extraction started')
        # Get raw energy
        with ProcessPoolExecutor(
                max_workers=self.num_workers) as executor, tqdm(
                    total=len(wav_list)) as progress:
            futures = []
            for wav_basename, pcm_data in pcm_dict.items():
                future = executor.submit(get_energy, pcm_data, self.hop_length,
                                         self.win_length, self.n_fft)
                future.add_done_callback(lambda p: progress.update())
                futures.append((future, wav_basename))

            for future, wav_basename in futures:
                result = future.result()
                if result is None:
                    logging.warning(
                        '[AudioProcessor] Energy extraction failed for %s',
                        wav_basename)
                    self.badcase_list.append(wav_basename)
                else:
                    energy = result
                    if len(mel_dict) > 0:
                        energy = align_length(energy,
                                              mel_dict.get(wav_basename, None))
                    if energy is None:
                        logging.warning(
                            '[AudioProcessor] Energy length mismatch with mel in %s',
                            wav_basename,
                        )
                        self.badcase_list.append(wav_basename)
                        continue
                    self.energy_dict[wav_basename] = energy

        logging.info('Melspec statistic proceeding...')
        #  Normalize energy
        energy_mean = compute_mean(list(self.energy_dict.values()), dims=1)
        energy_std = compute_std(
            list(self.energy_dict.values()), energy_mean, dims=1)
        np.savetxt(
            os.path.join(out_energy_dir, 'energy_mean.txt'),
            energy_mean,
            fmt='%.6f')
        np.savetxt(
            os.path.join(out_energy_dir, 'energy_std.txt'),
            energy_std,
            fmt='%.6f')
        logging.info(
            '[AudioProcessor] energy mean and std saved to:\n{},\n{}'.format(
                os.path.join(out_energy_dir, 'energy_mean.txt'),
                os.path.join(out_energy_dir, 'energy_std.txt'),
            ))

        logging.info('[AudioProcessor] Energy mean std norm is proceeding...')
        for wav_basename in self.energy_dict:
            energy = self.energy_dict[wav_basename]
            norm_energy = f0_norm_mean_std(energy, energy_mean, energy_std)
            self.energy_dict[wav_basename] = norm_energy

        #  save frame energy to a specific dir
        for wav_basename in self.energy_dict:
            np.save(
                os.path.join(out_frame_energy_dir, wav_basename + '.npy'),
                self.energy_dict[wav_basename].reshape(-1),
            )

        #  phone level average
        #  if there is no duration then save the frame-level energy
        if self.phone_level_feature and len(self.dur_dict) > 0:
            with ProcessPoolExecutor(
                    max_workers=self.num_workers) as executor, tqdm(
                        total=len(self.energy_dict)) as progress:
                futures = []
                for wav_basename in self.energy_dict:
                    future = executor.submit(
                        average_by_duration,
                        self.energy_dict.get(wav_basename, None),
                        self.dur_dict.get(wav_basename, None),
                    )
                    future.add_done_callback(lambda p: progress.update())
                    futures.append((future, wav_basename))

                for future, wav_basename in futures:
                    result = future.result()
                    if result is None:
                        logging.warning(
                            '[AudioProcessor] Energy extraction failed in phone level avg for: %s',
                            wav_basename,
                        )
                        self.badcase_list.append(wav_basename)
                    else:
                        avg_energy = result
                        self.energy_dict[wav_basename] = avg_energy

        for wav_basename in self.energy_dict:
            np.save(
                os.path.join(out_energy_dir, wav_basename + '.npy'),
                self.energy_dict[wav_basename].reshape(-1),
            )

        logging.info('[AudioProcessor] Energy normalization finished')
        logging.info('[AudioProcessor] Normed Energy saved to %s',
                     out_energy_dir)
        logging.info('[AudioProcessor] Energy extraction finished')

        return True

    def process(self, src_voice_dir, out_data_dir, aux_metafile=None):
        succeed = True

        raw_wav_dir = os.path.join(src_voice_dir, 'wav')
        src_interval_dir = os.path.join(src_voice_dir, 'interval')

        out_mel_dir = os.path.join(out_data_dir, 'mel')
        out_f0_dir = os.path.join(out_data_dir, 'f0')
        out_frame_f0_dir = os.path.join(out_data_dir, 'frame_f0')
        out_frame_uv_dir = os.path.join(out_data_dir, 'frame_uv')
        out_energy_dir = os.path.join(out_data_dir, 'energy')
        out_frame_energy_dir = os.path.join(out_data_dir, 'frame_energy')
        out_duration_dir = os.path.join(out_data_dir, 'raw_duration')
        out_cali_duration_dir = os.path.join(out_data_dir, 'duration')

        os.makedirs(out_data_dir, exist_ok=True)

        with_duration = os.path.exists(src_interval_dir)
        train_wav_dir = os.path.join(out_data_dir, 'wav')

        succeed = self.amp_normalize(raw_wav_dir, train_wav_dir)
        if not succeed:
            logging.error('[AudioProcessor] amp_normalize failed, exit')
            return False

        if with_duration:
            #  Raw duration, non-trimmed
            succeed = self.duration_generate(src_interval_dir,
                                             out_duration_dir)
            if not succeed:
                logging.error(
                    '[AudioProcessor] duration_generate failed, exit')
                return False

        if self.trim_silence:
            if with_duration:
                succeed = self.trim_silence_wav_with_interval(
                    train_wav_dir, out_duration_dir)
                if not succeed:
                    logging.error(
                        '[AudioProcessor] trim_silence_wav_with_interval failed, exit'
                    )
                    return False
            else:
                succeed = self.trim_silence_wav(train_wav_dir)
                if not succeed:
                    logging.error(
                        '[AudioProcessor] trim_silence_wav failed, exit')
                    return False

        succeed = self.mel_extract(train_wav_dir, out_mel_dir)
        if not succeed:
            logging.error('[AudioProcessor] mel_extract failed, exit')
            return False

        if aux_metafile is not None and with_duration:
            self.calibrate_SyllableDuration(out_duration_dir, aux_metafile,
                                            out_cali_duration_dir)

        succeed = self.pitch_extract(train_wav_dir, out_f0_dir,
                                     out_frame_f0_dir, out_frame_uv_dir)
        if not succeed:
            logging.error('[AudioProcessor] pitch_extract failed, exit')
            return False

        succeed = self.energy_extract(train_wav_dir, out_energy_dir,
                                      out_frame_energy_dir)
        if not succeed:
            logging.error('[AudioProcessor] energy_extract failed, exit')
            return False

        # recording badcase list
        with open(os.path.join(out_data_dir, 'badlist.txt'), 'w') as f:
            f.write('\n'.join(self.badcase_list))

        logging.info('[AudioProcessor] All features extracted successfully!')

        return succeed
