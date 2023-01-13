# Copyright (c) Alibaba, Inc. and its affiliates.

import csv
import os
from typing import Dict, Optional, Union

import numpy as np
import speechbrain as sb
import speechbrain.nnet.schedulers as schedulers
import torch
import torch.nn.functional as F
import torchaudio
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from tqdm import tqdm

from modelscope.metainfo import Trainers
from modelscope.models import Model, TorchModel
from modelscope.msdatasets import MsDataset
from modelscope.trainers.base import BaseTrainer
from modelscope.trainers.builder import TRAINERS
from modelscope.utils.constant import DEFAULT_MODEL_REVISION, ModelFile
from modelscope.utils.device import create_device
from modelscope.utils.logger import get_logger
from modelscope.utils.torch_utils import (get_dist_info, get_local_rank,
                                          init_dist)

EVAL_KEY = 'si-snr'

logger = get_logger()


@TRAINERS.register_module(module_name=Trainers.speech_separation)
class SeparationTrainer(BaseTrainer):
    """A trainer is used for speech separation.

    Args:
        model: id or local path of the model
        work_dir: local path to store all training outputs
        cfg_file: config file of the model
        train_dataset: dataset for training
        eval_dataset: dataset for evaluation
        model_revision: the git version of model on modelhub
    """

    def __init__(self,
                 model: str,
                 work_dir: str,
                 cfg_file: Optional[str] = None,
                 train_dataset: Optional[Union[MsDataset, Dataset]] = None,
                 eval_dataset: Optional[Union[MsDataset, Dataset]] = None,
                 model_revision: Optional[str] = DEFAULT_MODEL_REVISION,
                 **kwargs):

        if isinstance(model, str):
            self.model_dir = self.get_or_download_model_dir(
                model, model_revision)
            if cfg_file is None:
                cfg_file = os.path.join(self.model_dir,
                                        ModelFile.CONFIGURATION)
        else:
            assert cfg_file is not None, 'Config file should not be None if model is not from pretrained!'
            self.model_dir = os.path.dirname(cfg_file)

        BaseTrainer.__init__(self, cfg_file)

        self.model = self.build_model()
        self.work_dir = work_dir
        if kwargs.get('launcher', None) is not None:
            init_dist(kwargs['launcher'])
        _, world_size = get_dist_info()
        self._dist = world_size > 1

        device_name = kwargs.get('device', 'gpu')
        if self._dist:
            local_rank = get_local_rank()
            device_name = f'cuda:{local_rank}'
        self.device = create_device(device_name)

        if 'max_epochs' not in kwargs:
            assert hasattr(
                self.cfg.train, 'max_epochs'
            ), 'max_epochs is missing from the configuration file'
            self._max_epochs = self.cfg.train.max_epochs
        else:
            self._max_epochs = kwargs['max_epochs']
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        hparams_file = os.path.join(self.model_dir, 'hparams.yaml')
        overrides = {
            'output_folder':
            self.work_dir,
            'seed':
            self.cfg.train.seed,
            'lr':
            self.cfg.train.optimizer.lr,
            'weight_decay':
            self.cfg.train.optimizer.weight_decay,
            'clip_grad_norm':
            self.cfg.train.optimizer.clip_grad_norm,
            'factor':
            self.cfg.train.lr_scheduler.factor,
            'patience':
            self.cfg.train.lr_scheduler.patience,
            'dont_halve_until_epoch':
            self.cfg.train.lr_scheduler.dont_halve_until_epoch,
        }
        # load hyper params
        from hyperpyyaml import load_hyperpyyaml
        with open(hparams_file) as fin:
            self.hparams = load_hyperpyyaml(fin, overrides=overrides)
        # Create experiment directory
        sb.create_experiment_directory(
            experiment_directory=self.work_dir,
            hyperparams_to_save=hparams_file,
            overrides=overrides,
        )

        run_opts = {
            'debug': False,
            'device': 'cpu',
            'data_parallel_backend': False,
            'distributed_launch': False,
            'distributed_backend': 'nccl',
            'find_unused_parameters': False
        }
        if self.device.type == 'cuda':
            run_opts['device'] = f'{self.device.type}:{self.device.index}'
        self.epoch_counter = sb.utils.epoch_loop.EpochCounter(self._max_epochs)
        self.hparams['epoch_counter'] = self.epoch_counter
        self.hparams['checkpointer'].add_recoverables(
            {'counter': self.epoch_counter})
        modules = self.model.as_dict()
        self.hparams['checkpointer'].add_recoverables(modules)
        # Brain class initialization
        self.separator = Separation(
            modules=modules,
            opt_class=self.hparams['optimizer'],
            hparams=self.hparams,
            run_opts=run_opts,
            checkpointer=self.hparams['checkpointer'],
        )

    def build_model(self) -> torch.nn.Module:
        """ Instantiate a pytorch model and return.
        """
        model = Model.from_pretrained(
            self.model_dir, cfg_dict=self.cfg, training=True)
        if isinstance(model, TorchModel) and hasattr(model, 'model'):
            return model.model
        elif isinstance(model, torch.nn.Module):
            return model

    def train(self, *args, **kwargs):
        self.separator.fit(
            self.epoch_counter,
            self.train_dataset,
            self.eval_dataset,
            train_loader_kwargs=self.hparams['dataloader_opts'],
            valid_loader_kwargs=self.hparams['dataloader_opts'],
        )

    def evaluate(self, checkpoint_path: str, *args,
                 **kwargs) -> Dict[str, float]:
        if checkpoint_path:
            self.hparams.checkpointer.checkpoints_dir = checkpoint_path
        else:
            self.model.load_check_point(device=self.device)
        value = self.separator.evaluate(
            self.eval_dataset,
            test_loader_kwargs=self.hparams['dataloader_opts'],
            min_key=EVAL_KEY)
        return {EVAL_KEY: value}


class Separation(sb.Brain):
    """A subclass of speechbrain.Brain implements training steps."""

    def compute_forward(self, mix, targets, stage, noise=None):
        """Forward computations from the mixture to the separated signals."""

        # Unpack lists and put tensors in the right device
        mix, mix_lens = mix
        mix, mix_lens = mix.to(self.device), mix_lens.to(self.device)

        # Convert targets to tensor
        targets = torch.cat(
            [
                targets[i][0].unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        ).to(self.device)

        # Add speech distortions
        if stage == sb.Stage.TRAIN:
            with torch.no_grad():
                if self.hparams.use_speedperturb or self.hparams.use_rand_shift:
                    mix, targets = self.add_speed_perturb(targets, mix_lens)

                    mix = targets.sum(-1)

                if self.hparams.use_wavedrop:
                    mix = self.hparams.wavedrop(mix, mix_lens)

                if self.hparams.limit_training_signal_len:
                    mix, targets = self.cut_signals(mix, targets)

        # Separation
        mix_w = self.modules['encoder'](mix)
        est_mask = self.modules['masknet'](mix_w)
        mix_w = torch.stack([mix_w] * self.hparams.num_spks)
        sep_h = mix_w * est_mask

        # Decoding
        est_source = torch.cat(
            [
                self.modules['decoder'](sep_h[i]).unsqueeze(-1)
                for i in range(self.hparams.num_spks)
            ],
            dim=-1,
        )
        # T changed after conv1d in encoder, fix it here
        T_origin = mix.size(1)
        T_est = est_source.size(1)
        if T_origin > T_est:
            est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets

    def compute_objectives(self, predictions, targets):
        """Computes the sinr loss"""
        return self.hparams.loss(targets, predictions)

    # yapf: disable
    def fit_batch(self, batch):
        """Trains one batch"""
        # Unpacking batch list
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]

        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        if self.auto_mix_prec:
            with autocast():
                predictions, targets = self.compute_forward(
                    mixture, targets, sb.Stage.TRAIN)
                loss = self.compute_objectives(predictions, targets)
                # hard threshold the easy dataitems
                if self.hparams.threshold_byloss:
                    th = self.hparams.threshold
                    loss_to_keep = loss[loss > th]
                    if loss_to_keep.nelement() > 0:
                        loss = loss_to_keep.mean()
                    else:
                        print('loss has zero elements!!')
                else:
                    loss = loss.mean()

            # the fix for computational problems
            if loss < self.hparams.loss_upper_lim and loss.nelement() > 0:
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(),
                        self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    'infinite loss or empty loss! it happened {} times so far - skipping this batch'
                    .format(self.nonfinite_count))
                loss.data = torch.tensor(0).to(self.device)
        else:
            predictions, targets = self.compute_forward(
                mixture, targets, sb.Stage.TRAIN)
            loss = self.compute_objectives(predictions, targets)
            if self.hparams.threshold_byloss:
                th = self.hparams.threshold
                loss_to_keep = loss[loss > th]
                if loss_to_keep.nelement() > 0:
                    loss = loss_to_keep.mean()
            else:
                loss = loss.mean()
            # the fix for computational problems
            if loss < self.hparams.loss_upper_lim and loss.nelement() > 0:
                loss.backward()
                if self.hparams.clip_grad_norm >= 0:
                    torch.nn.utils.clip_grad_norm_(self.modules.parameters(),
                                                   self.hparams.clip_grad_norm)
                self.optimizer.step()
            else:
                self.nonfinite_count += 1
                logger.info(
                    'infinite loss or empty loss! it happened {} times so far - skipping this batch'
                    .format(self.nonfinite_count))
                loss.data = torch.tensor(0).to(self.device)
        self.optimizer.zero_grad()
        return loss.detach().cpu()
    # yapf: enable

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        snt_id = batch.id
        mixture = batch.mix_sig
        targets = [batch.s1_sig, batch.s2_sig]
        if self.hparams.num_spks == 3:
            targets.append(batch.s3_sig)

        with torch.no_grad():
            predictions, targets = self.compute_forward(
                mixture, targets, stage)
            loss = self.compute_objectives(predictions, targets)

        # Manage audio file saving
        if stage == sb.Stage.TEST and self.hparams.save_audio:
            if hasattr(self.hparams, 'n_audio_to_save'):
                if self.hparams.n_audio_to_save > 0:
                    self.save_audio(snt_id[0], mixture, targets, predictions)
                    self.hparams.n_audio_to_save += -1
            else:
                self.save_audio(snt_id[0], mixture, targets, predictions)

        return loss.mean().detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {'si-snr': stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            # Learning rate annealing
            if isinstance(self.hparams.lr_scheduler,
                          schedulers.ReduceLROnPlateau):
                current_lr, next_lr = self.hparams.lr_scheduler(
                    [self.optimizer], epoch, stage_loss)
                schedulers.update_learning_rate(self.optimizer, next_lr)
            else:
                # if we do not use the reducelronplateau, we do not change the lr
                current_lr = self.hparams.optimizer.optim.param_groups[0]['lr']

            self.hparams.train_logger.log_stats(
                stats_meta={
                    'epoch': epoch,
                    'lr': current_lr
                },
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={'si-snr': stage_stats['si-snr']},
                min_keys=['si-snr'],
            )

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        if self.hparams.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True

            for i in range(targets.shape[-1]):
                new_target = self.hparams.speedperturb(targets[:, :, i],
                                                       targ_lens)
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            if self.hparams.use_rand_shift:
                # Performing random_shift (independently on each source)
                recombine = True
                for i in range(targets.shape[-1]):
                    rand_shift = torch.randint(self.hparams.min_shift,
                                               self.hparams.max_shift, (1, ))
                    new_targets[i] = new_targets[i].to(self.device)
                    new_targets[i] = torch.roll(
                        new_targets[i], shifts=(rand_shift[0], ), dims=1)

            # Re-combination
            if recombine:
                if self.hparams.use_speedperturb:
                    targets = torch.zeros(
                        targets.shape[0],
                        min_len,
                        targets.shape[-1],
                        device=targets.device,
                        dtype=torch.float,
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets

    def cut_signals(self, mixture, targets):
        """This function selects a random segment of a given length within the mixture.
        The corresponding targets are selected accordingly"""
        randstart = torch.randint(
            0,
            1 + max(0, mixture.shape[1] - self.hparams.training_signal_len),
            (1, ),
        ).item()
        targets = targets[:, randstart:randstart
                          + self.hparams.training_signal_len, :]
        mixture = mixture[:, randstart:randstart
                          + self.hparams.training_signal_len]
        return mixture, targets

    def reset_layer_recursively(self, layer):
        """Reinitializes the parameters of the neural networks"""
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
        for child_layer in layer.modules():
            if layer != child_layer:
                self.reset_layer_recursively(child_layer)

    def save_results(self, test_data):
        """This script computes the SDR and SI-SNR metrics and saves
        them into a csv file"""

        # This package is required for SDR computation
        from mir_eval.separation import bss_eval_sources

        # Create folders where to store audio
        save_file = os.path.join(self.hparams.output_folder,
                                 'test_results.csv')

        # Variable init
        all_sdrs = []
        all_sdrs_i = []
        all_sisnrs = []
        all_sisnrs_i = []
        csv_columns = ['snt_id', 'sdr', 'sdr_i', 'si-snr', 'si-snr_i']

        test_loader = sb.dataio.dataloader.make_dataloader(
            test_data, **self.hparams.dataloader_opts)

        with open(save_file, 'w') as results_csv:
            writer = csv.DictWriter(results_csv, fieldnames=csv_columns)
            writer.writeheader()

            # Loop over all test sentence
            with tqdm(test_loader, dynamic_ncols=True) as t:
                for i, batch in enumerate(t):

                    # Apply Separation
                    mixture, mix_len = batch.mix_sig
                    snt_id = batch.id
                    targets = [batch.s1_sig, batch.s2_sig]
                    if self.hparams.num_spks == 3:
                        targets.append(batch.s3_sig)

                    with torch.no_grad():
                        predictions, targets = self.compute_forward(
                            batch.mix_sig, targets, sb.Stage.TEST)

                    # Compute SI-SNR
                    sisnr = self.compute_objectives(predictions, targets)

                    # Compute SI-SNR improvement
                    mixture_signal = torch.stack(
                        [mixture] * self.hparams.num_spks, dim=-1)
                    mixture_signal = mixture_signal.to(targets.device)
                    sisnr_baseline = self.compute_objectives(
                        mixture_signal, targets)
                    sisnr_i = sisnr.mean() - sisnr_baseline.mean()

                    # Compute SDR
                    sdr, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        predictions[0].t().detach().cpu().numpy(),
                    )

                    sdr_baseline, _, _, _ = bss_eval_sources(
                        targets[0].t().cpu().numpy(),
                        mixture_signal[0].t().detach().cpu().numpy(),
                    )

                    sdr_i = sdr.mean() - sdr_baseline.mean()

                    # Saving on a csv file
                    row = {
                        'snt_id': snt_id[0],
                        'sdr': sdr.mean(),
                        'sdr_i': sdr_i,
                        'si-snr': -sisnr.item(),
                        'si-snr_i': -sisnr_i.item(),
                    }
                    writer.writerow(row)

                    # Metric Accumulation
                    all_sdrs.append(sdr.mean())
                    all_sdrs_i.append(sdr_i.mean())
                    all_sisnrs.append(-sisnr.item())
                    all_sisnrs_i.append(-sisnr_i.item())

                row = {
                    'snt_id': 'avg',
                    'sdr': np.array(all_sdrs).mean(),
                    'sdr_i': np.array(all_sdrs_i).mean(),
                    'si-snr': np.array(all_sisnrs).mean(),
                    'si-snr_i': np.array(all_sisnrs_i).mean(),
                }
                writer.writerow(row)

        logger.info('Mean SISNR is {}'.format(np.array(all_sisnrs).mean()))
        logger.info('Mean SISNRi is {}'.format(np.array(all_sisnrs_i).mean()))
        logger.info('Mean SDR is {}'.format(np.array(all_sdrs).mean()))
        logger.info('Mean SDRi is {}'.format(np.array(all_sdrs_i).mean()))

    def save_audio(self, snt_id, mixture, targets, predictions):
        'saves the test audio (mixture, targets, and estimated sources) on disk'

        # Create outout folder
        save_path = os.path.join(self.hparams.save_folder, 'audio_results')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for ns in range(self.hparams.num_spks):

            # Estimated source
            signal = predictions[0, :, ns]
            signal = signal / signal.abs().max() * 0.5
            save_file = os.path.join(
                save_path, 'item{}_source{}hat.wav'.format(snt_id, ns + 1))
            torchaudio.save(save_file,
                            signal.unsqueeze(0).cpu(),
                            self.hparams.sample_rate)

            # Original source
            signal = targets[0, :, ns]
            signal = signal / signal.abs().max() * 0.5
            save_file = os.path.join(
                save_path, 'item{}_source{}.wav'.format(snt_id, ns + 1))
            torchaudio.save(save_file,
                            signal.unsqueeze(0).cpu(),
                            self.hparams.sample_rate)

        # Mixture
        signal = mixture[0][0, :]
        signal = signal / signal.abs().max() * 0.5
        save_file = os.path.join(save_path, 'item{}_mix.wav'.format(snt_id))
        torchaudio.save(save_file,
                        signal.unsqueeze(0).cpu(), self.hparams.sample_rate)
