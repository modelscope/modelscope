# Copyright (c) Alibaba, Inc. and its affiliates.

import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from modelscope.models.audio.tts.kantts.utils.plot import (plot_alignment,
                                                           plot_spectrogram)
from modelscope.utils.logger import get_logger

logging = get_logger()


def traversal_dict(d, func):
    if not isinstance(d, dict):
        logging.error('Not a dict: {}'.format(d))
        return
    for k, v in d.items():
        if isinstance(v, dict):
            traversal_dict(v, func)
        else:
            func(k, v)


def distributed_init():
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('RANK', 0))
    distributed = world_size > 1
    device = torch.device('cuda', local_rank)
    if distributed:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        logging.info(
            'Distributed training, global world size: {}, local world size: {}, global rank: {}, local rank: {}'
            .format(
                world_size,
                torch.cuda.device_count(),
                torch.distributed.get_rank(),
                local_rank,
            ))
        logging.info('nccl backend: {}'.format(
            torch.distributed.is_nccl_available()))
        logging.info('mpi backend: {}'.format(
            torch.distributed.is_mpi_available()))
        device_ids = list(range(torch.cuda.device_count()))
        logging.info(
            '[{}] rank = {}, world_size = {}, n_gpus = {}, device_ids = {}'.
            format(
                os.getpid(),
                torch.distributed.get_rank(),
                torch.distributed.get_world_size(),
                torch.cuda.device_count(),
                device_ids,
            ))
    return distributed, device, local_rank, world_size


class Trainer(object):

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        sampler,
        train_loader,
        valid_loader,
        max_epochs=None,
        max_steps=None,
        save_dir=None,
        save_interval=1,
        valid_interval=1,
        log_interval=10,
        grad_clip=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.sampler = sampler
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.max_epochs = max_epochs
        self.steps = 1
        self.epoch = 0
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.valid_interval = valid_interval
        self.log_interval = log_interval
        self.grad_clip = grad_clip
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.config = config
        self.distributed = self.config.get('distributed', False)
        self.rank = self.config.get('rank', 0)

        self.log_dir = os.path.join(save_dir, 'log')
        self.ckpt_dir = os.path.join(save_dir, 'ckpt')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        if max_epochs is None:
            self.max_epochs = sys.maxsize
        else:
            self.max_epochs = int(max_epochs)
        if max_steps is None:
            self.max_steps = sys.maxsize
        else:
            self.max_steps = int(max_steps)

        self.finish_training = False

    def set_model_state(self, state='train'):
        if state == 'train':
            if isinstance(self.model, dict):
                for key in self.model.keys():
                    self.model[key].train()
            else:
                self.model.train()
        elif state == 'eval':
            if isinstance(self.model, dict):
                for key in self.model.keys():
                    self.model[key].eval()
            else:
                self.model.eval()
        else:
            raise ValueError("state must be either 'train' or 'eval'.")

    def write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'steps': self.steps,
            'model': self.model.state_dict(),
        }

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self,
                        checkpoint_path,
                        restore_training_state=False,
                        strict=True):
        state_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(state_dict['model'], strict=strict)
        if restore_training_state:
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.scheduler.load_state_dict(state_dict['scheduler'])
            self.steps = state_dict['steps']

    def check_save_interval(self):
        if self.ckpt_dir is not None and (
                self.steps) % self.save_interval == 0:
            self.save_checkpoint(
                os.path.join(self.ckpt_dir,
                             'checkpoint_{}.pth'.format(self.steps)))
            logging.info('Checkpoint saved at step {}'.format(self.steps))

    def check_log_interval(self):
        if self.writer is not None and (self.steps) % self.log_interval == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config['log_interval_steps']
                logging.info(
                    f'(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}.'
                )
            self.write_to_tensorboard(self.total_train_loss)
            self.total_train_loss = defaultdict(float)

            def log_learning_rate(key, sche):
                logging.info('{} learning rate: {:.6f}'.format(
                    key,
                    sche.get_lr()[0]))
                self.write_to_tensorboard(
                    {'{}_lr'.format(key): sche.get_lr()[0]})

            traversal_dict(self.scheduler, log_learning_rate)

    def check_eval_interval(self):
        if self.valid_interval > 0 and (self.steps) % self.valid_interval == 0:
            self.eval_epoch()

    def check_stop_training(self):
        if self.steps >= self.max_steps or self.epoch >= self.max_epochs:
            self.finish_training = True

    def train(self):
        self.set_model_state('train')

        while True:
            self.train_epoch()
            self.epoch += 1
            self.check_stop_training()
            if self.finish_training:
                break

    def train_epoch(self):
        for batch in tqdm(self.train_loader):
            self.train_step(batch)

            if self.rank == 0:
                self.check_eval_interval()
                self.check_save_interval()
                self.check_log_interval()

            self.steps += 1
            self.check_stop_training()
            if self.finish_training:
                break

        logging.info('Epoch {} finished'.format(self.epoch))

        if self.distributed:
            self.sampler['train'].set_epoch(self.epoch)

    def train_step(self, batch):
        data, target = batch
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optimizer.step()

    @torch.no_grad()
    def eval_step(self, batch):
        pass

    def eval_epoch(self):
        logging.info(f'(Epoch: {self.epoch}) Start evaluation.')
        # change mode
        self.set_model_state('eval')

        self.total_eval_loss = defaultdict(float)
        rand_idx = np.random.randint(0, len(self.valid_loader))
        idx = 0
        logging.info('Valid data size: {}'.format(len(self.valid_loader)))
        for batch in tqdm(self.valid_loader):
            self.eval_step(batch)
            if idx == rand_idx:
                logging.info(
                    f'(Epoch: {self.epoch}) Random batch: {idx}, generating image.'
                )
                self.genearete_and_save_intermediate_result(batch)
            idx += 1

        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= idx + 1
            logging.info(
                f'(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}.'
            )
        self.write_to_tensorboard(self.total_eval_loss)

        logging.info('Epoch {} evaluation finished'.format(self.epoch))

        self.set_model_state('train')

    @torch.no_grad()
    def genearete_and_save_intermediate_result(self, batch):
        pass


class GAN_Trainer(Trainer):

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        sampler,
        train_loader,
        valid_loader,
        max_epochs=None,
        max_steps=None,
        save_dir=None,
        save_interval=1,
        valid_interval=1,
        log_interval=10,
        grad_clip=None,
    ):
        super().__init__(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            sampler,
            train_loader,
            valid_loader,
            max_epochs,
            max_steps,
            save_dir,
            save_interval,
            valid_interval,
            log_interval,
            grad_clip,
        )

    def set_model_state(self, state='train'):
        if state == 'train':
            if isinstance(self.model, dict):
                self.model['generator'].train()
                for key in self.model['discriminator'].keys():
                    self.model['discriminator'][key].train()
            else:
                self.model.train()
        elif state == 'eval':
            if isinstance(self.model, dict):
                self.model['generator'].eval()
                for key in self.model['discriminator'].keys():
                    self.model['discriminator'][key].eval()
            else:
                self.model.eval()
        else:
            raise ValueError("state must be either 'train' or 'eval'.")

    @torch.no_grad()
    def genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # generate
        y_batch, x_batch = batch
        y_batch, x_batch = y_batch.to(self.device), x_batch.to(self.device)
        y_batch_ = self.model['generator'](x_batch)
        if self.model.get('pqmf', None):
            y_mb_ = y_batch_
            y_batch_ = self.model['pqmf'].synthesis(y_mb_)

        # check directory
        dirname = os.path.join(self.log_dir, f'predictions/{self.steps}steps')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for idx, (y, y_) in enumerate(zip(y_batch, y_batch_), 1):
            # convert to ndarray
            y, y_ = y.view(-1).cpu().numpy(), y_.view(-1).cpu().numpy()

            # plot figure and save it
            figname = os.path.join(dirname, f'{idx}.png')
            plt.subplot(2, 1, 1)
            plt.plot(y)
            plt.title('groundtruth speech')
            plt.subplot(2, 1, 2)
            plt.plot(y_)
            plt.title(f'generated speech @ {self.steps} steps')
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # save as wavfile
            y = np.clip(y, -1, 1)
            y_ = np.clip(y_, -1, 1)
            sf.write(
                figname.replace('.png', '_ref.wav'),
                y,
                self.config['audio_config']['sampling_rate'],
                'PCM_16',
            )
            sf.write(
                figname.replace('.png', '_gen.wav'),
                y_,
                self.config['audio_config']['sampling_rate'],
                'PCM_16',
            )

            if idx >= self.config['num_save_intermediate_results']:
                break

    @torch.no_grad()
    def eval_step(self, batch):
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)

        y_ = self.model['generator'](x)
        # reconstruct the signal from multi-band signal
        if self.model.get('pqmf', None):
            y_mb_ = y_
            y_ = self.model['pqmf'].synthesis(y_mb_)

        aux_loss = 0.0

        # multi-resolution sfft loss
        if self.criterion.get('stft_loss', None):
            sc_loss, mag_loss = self.criterion['stft_loss'](y_, y)
            aux_loss += (sc_loss
                         + mag_loss) * self.criterion['stft_loss'].weights
            self.total_eval_loss[
                'eval/spectral_convergence_loss'] += sc_loss.item()

        # subband multi-resolution stft loss
        if self.criterion.get('subband_stft_loss', None):
            aux_loss *= 0.5  # for balancing with subband stft loss
            y_mb = self.model['pqmf'].analysis(y)
            sub_sc_loss, sub_mag_loss = self.criterion['sub_stft'](y_mb_, y_mb)
            self.total_eval_loss[
                'eval/sub_spectral_convergence_loss'] += sub_sc_loss.item()
            self.total_eval_loss[
                'eval/sub_log_stft_magnitude_loss'] += sub_mag_loss.item()
            aux_loss += (0.5 * (sub_sc_loss + sub_mag_loss)
                         * self.criterion['sub_stft'].weights)

        # mel spectrogram loss
        if self.criterion.get('mel_loss', None):
            mel_loss = self.criterion['mel_loss'](y_, y)
            aux_loss += mel_loss * self.criterion['mel_loss'].weights
            self.total_eval_loss['eval/mel_loss'] += mel_loss.item()

        fmap_lst_ = []
        adv_loss = 0.0
        #  adversiral loss
        for discriminator in self.model['discriminator'].keys():
            p_, fmap_ = self.model['discriminator'][discriminator](y_)
            fmap_lst_.append(fmap_)
            adv_loss += (
                self.criterion['generator_adv_loss'](p_)
                * self.criterion['generator_adv_loss'].weights)

        gen_loss = aux_loss + adv_loss

        if self.criterion.get('feat_match_loss', None):
            fmap_lst = []
            # no need to track gradients
            for discriminator in self.model['discriminator'].keys():
                with torch.no_grad():
                    p, fmap = self.model['discriminator'][discriminator](y)
                    fmap_lst.append(fmap)

            fm_loss = 0.0
            for fmap_, fmap in zip(fmap_lst, fmap_lst_):
                fm_loss += self.criterion['feat_match_loss'](fmap_, fmap)
            self.total_eval_loss['eval/feature_matching_loss'] += fm_loss.item(
            )

            gen_loss += fm_loss * self.criterion['feat_match_loss'].weights

        dis_loss = 0.0
        for discriminator in self.model['discriminator'].keys():
            p, fmap = self.model['discriminator'][discriminator](y)
            p_, fmap_ = self.model['discriminator'][discriminator](y_.detach())
            real_loss, fake_loss = self.criterion['discriminator_adv_loss'](p_,
                                                                            p)
            dis_loss += real_loss + fake_loss
            self.total_eval_loss['eval/real_loss'] += real_loss.item()
            self.total_eval_loss['eval/fake_loss'] += fake_loss.item()

        self.total_eval_loss['eval/discriminator_loss'] += dis_loss.item()
        self.total_eval_loss['eval/adversarial_loss'] += adv_loss.item()
        self.total_eval_loss['eval/generator_loss'] += gen_loss.item()

    def train_step(self, batch):
        y, x = batch
        y, x = y.to(self.device), x.to(self.device)

        if self.steps >= self.config.get('generator_train_start_steps', 0):
            y_ = self.model['generator'](x)
            # reconstruct the signal from multi-band signal
            if self.model.get('pqmf', None):
                y_mb_ = y_
                y_ = self.model['pqmf'].synthesis(y_mb_)

            # initialize
            gen_loss = 0.0

            # multi-resolution sfft loss
            if self.criterion.get('stft_loss', None):
                sc_loss, mag_loss = self.criterion['stft_loss'](y_, y)
                gen_loss += (sc_loss
                             + mag_loss) * self.criterion['stft_loss'].weights
                self.total_train_loss[
                    'train/spectral_convergence_loss'] += sc_loss.item()
                self.total_train_loss[
                    'train/log_stft_magnitude_loss'] += mag_loss.item()

            # subband multi-resolution stft loss
            if self.criterion.get('subband_stft_loss', None):
                gen_loss *= 0.5  # for balancing with subband stft loss
                y_mb = self.model['pqmf'].analysis(y)
                sub_sc_loss, sub_mag_loss = self.criterion['sub_stft'](y_mb_,
                                                                       y_mb)
                gen_loss += 0.5 * (sub_sc_loss + sub_mag_loss)
                self.total_train_loss[
                    'train/sub_spectral_convergence_loss'] += sub_sc_loss.item(
                    )  # noqa E123
                self.total_train_loss[
                    'train/sub_log_stft_magnitude_loss'] += sub_mag_loss.item(
                    )  # noqa E123

            # mel spectrogram loss
            if self.criterion.get('mel_loss', None):
                mel_loss = self.criterion['mel_loss'](y_, y)
                gen_loss += mel_loss * self.criterion['mel_loss'].weights
                self.total_train_loss['train/mel_loss'] += mel_loss.item()

            # adversarial loss
            if self.steps > self.config['discriminator_train_start_steps']:
                adv_loss = 0.0
                fmap_lst_ = []
                for discriminator in self.model['discriminator'].keys():
                    p_, fmap_ = self.model['discriminator'][discriminator](y_)
                    fmap_lst_.append(fmap_)
                    adv_loss += self.criterion['generator_adv_loss'](p_)
                    self.total_train_loss[
                        'train/adversarial_loss'] += adv_loss.item()

                gen_loss += adv_loss * self.criterion[
                    'generator_adv_loss'].weights

                # feature matching loss
                if self.criterion.get('feat_match_loss', None):
                    fmap_lst = []
                    # no need to track gradients
                    for discriminator in self.model['discriminator'].keys():
                        with torch.no_grad():
                            p, fmap = self.model['discriminator'][
                                discriminator](
                                    y)
                            fmap_lst.append(fmap)

                    fm_loss = 0.0
                    for fmap_, fmap in zip(fmap_lst, fmap_lst_):
                        fm_loss += self.criterion['feat_match_loss'](fmap_,
                                                                     fmap)
                    self.total_train_loss[
                        'train/feature_matching_loss'] += fm_loss.item()
                    gen_loss += fm_loss * self.criterion[
                        'feat_match_loss'].weights

            self.total_train_loss['train/generator_loss'] += gen_loss.item()
            # update generator
            self.optimizer['generator'].zero_grad()
            gen_loss.backward()
            if self.config['generator_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model['generator'].parameters(),
                    self.config['generator_grad_norm'],
                )
            self.optimizer['generator'].step()
            self.scheduler['generator'].step()

        # update discriminator
        if self.steps > self.config['discriminator_train_start_steps']:
            # re-compute y_ which leads better quality
            with torch.no_grad():
                y_ = self.model['generator'](x)

            if self.model.get('pqmf', None):
                y_ = self.model['pqmf'].synthesis(y_)

            # discriminator loss
            dis_loss = 0.0
            for discriminator in self.model['discriminator'].keys():
                p, fmap = self.model['discriminator'][discriminator](y)
                p_, fmap_ = self.model['discriminator'][discriminator](
                    y_.detach())
                real_loss, fake_loss = self.criterion[
                    'discriminator_adv_loss'](p_, p)
                dis_loss += real_loss + fake_loss
                self.total_train_loss['train/real_loss'] += real_loss.item()
                self.total_train_loss['train/fake_loss'] += fake_loss.item()

            self.total_train_loss['train/discriminator_loss'] += dis_loss.item(
            )

            # update discriminator
            for key in self.optimizer['discriminator'].keys():
                self.optimizer['discriminator'][key].zero_grad()

            dis_loss.backward()
            if self.config['discriminator_grad_norm'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model['discriminator'].parameters(),
                    self.config['discriminator_grad_norm'],
                )
            for key in self.optimizer['discriminator'].keys():
                self.optimizer['discriminator'][key].step()
            for key in self.scheduler['discriminator'].keys():
                self.scheduler['discriminator'][key].step()

    def save_checkpoint(self, checkpoint_path):
        state_dict = {
            'optimizer': {
                'generator': self.optimizer['generator'].state_dict(),
                'discriminator': {},
            },
            'scheduler': {
                'generator': self.scheduler['generator'].state_dict(),
                'discriminator': {},
            },
            'steps': self.steps,
        }
        for model_name in self.optimizer['discriminator'].keys():
            state_dict['optimizer']['discriminator'][
                model_name] = self.optimizer['discriminator'][
                    model_name].state_dict()

        for model_name in self.scheduler['discriminator'].keys():
            state_dict['scheduler']['discriminator'][
                model_name] = self.scheduler['discriminator'][
                    model_name].state_dict()

        if not self.distributed:
            model_state = self.model['generator'].state_dict()
        else:
            model_state = self.model['generator'].module.state_dict()
        state_dict['model'] = {
            'generator': model_state,
            'discriminator': {},
        }
        for model_name in self.model['discriminator'].keys():
            if not self.distributed:
                model_state = self.model['discriminator'][
                    model_name].state_dict()
            else:
                model_state = self.model['discriminator'][
                    model_name].module.state_dict()
            state_dict['model']['discriminator'][model_name] = model_state

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self,
                        checkpoint_path,
                        restore_training_state=False,
                        strict=True):
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        if not self.distributed:
            self.model['generator'].load_state_dict(
                state_dict['model']['generator'], strict=strict)
        else:
            self.model['generator'].module.load_state_dict(
                state_dict['model']['generator'], strict=strict)
        for model_name in state_dict['model']['discriminator']:
            if not self.distributed:
                self.model['discriminator'][model_name].load_state_dict(
                    state_dict['model']['discriminator'][model_name],
                    strict=strict)
            else:
                self.model['discriminator'][model_name].module.load_state_dict(
                    state_dict['model']['discriminator'][model_name],
                    strict=strict)

        if restore_training_state:
            self.steps = state_dict['steps']
            self.optimizer['generator'].load_state_dict(
                state_dict['optimizer']['generator'])
            self.scheduler['generator'].load_state_dict(
                state_dict['scheduler']['generator'])
            for model_name in state_dict['optimizer']['discriminator'].keys():
                self.optimizer['discriminator'][model_name].load_state_dict(
                    state_dict['optimizer']['discriminator'][model_name])
            for model_name in state_dict['scheduler']['discriminator'].keys():
                self.scheduler['discriminator'][model_name].load_state_dict(
                    state_dict['scheduler']['discriminator'][model_name])


class Sambert_Trainer(Trainer):

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        sampler,
        train_loader,
        valid_loader,
        max_epochs=None,
        max_steps=None,
        save_dir=None,
        save_interval=1,
        valid_interval=1,
        log_interval=10,
        grad_clip=None,
    ):
        super().__init__(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            sampler,
            train_loader,
            valid_loader,
            max_epochs,
            max_steps,
            save_dir,
            save_interval,
            valid_interval,
            log_interval,
            grad_clip,
        )
        self.with_MAS = config['Model']['KanTtsSAMBERT']['params'].get(
            'MAS', False)
        self.fp_enable = config['Model']['KanTtsSAMBERT']['params'].get(
            'FP', False)

    @torch.no_grad()
    def genearete_and_save_intermediate_result(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        inputs_emotion = batch['input_emotions'].to(self.device)
        inputs_speaker = batch['input_speakers'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        mel_targets = batch['mel_targets'].to(self.device)
        # generate mel spectrograms
        res = self.model['KanTtsSAMBERT'](
            inputs_ling[0:1],
            inputs_emotion[0:1],
            inputs_speaker[0:1],
            valid_input_lengths[0:1],
        )
        x_band_width = res['x_band_width']
        h_band_width = res['h_band_width']
        enc_slf_attn_lst = res['enc_slf_attn_lst']
        pnca_x_attn_lst = res['pnca_x_attn_lst']
        pnca_h_attn_lst = res['pnca_h_attn_lst']
        dec_outputs = res['dec_outputs']
        postnet_outputs = res['postnet_outputs']

        dirname = os.path.join(self.log_dir, f'predictions/{self.steps}steps')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for layer_id, slf_attn in enumerate(enc_slf_attn_lst):
            for head_id in range(self.config['Model']['KanTtsSAMBERT']
                                 ['params']['encoder_num_heads']):
                fig = plot_alignment(
                    slf_attn[head_id, :valid_input_lengths[0], :
                             valid_input_lengths[0]].cpu().numpy(),
                    info='valid_len_{}'.format(valid_input_lengths[0].item()),
                )
                fig.savefig(
                    os.path.join(
                        dirname,
                        'enc_slf_attn_dev_layer{}_head{}'.format(
                            layer_id, head_id),
                    ))
        for layer_id, (pnca_x_attn, pnca_h_attn) in enumerate(
                zip(pnca_x_attn_lst, pnca_h_attn_lst)):
            for head_id in range(self.config['Model']['KanTtsSAMBERT']
                                 ['params']['decoder_num_heads']):
                fig = plot_alignment(
                    pnca_x_attn[head_id, :, :].cpu().numpy(),
                    info='x_band_width_{}'.format(x_band_width),
                )
                fig.savefig(
                    os.path.join(
                        dirname,
                        'pnca_x_attn_dev_layer{}_head{}'.format(
                            layer_id, head_id),
                    ))
                fig = plot_alignment(
                    pnca_h_attn[head_id, :, :].cpu().numpy(),
                    info='h_band_width_{}'.format(h_band_width),
                )
                fig.savefig(
                    os.path.join(
                        dirname,
                        'pnca_h_attn_dev_layer{}_head{}'.format(
                            layer_id, head_id),
                    ))

        target_mel = mel_targets[0].cpu().numpy()
        coarse_mel = dec_outputs.squeeze(0).cpu().numpy()
        output_mel = postnet_outputs.squeeze(0).cpu().numpy()
        np.save(os.path.join(dirname, 'coarse_mel.npy'), coarse_mel)
        np.save(os.path.join(dirname, 'output_mel.npy'), output_mel)
        np.save(os.path.join(dirname, 'target_mel.npy'), target_mel)
        fig = plot_spectrogram(coarse_mel.T)
        fig.savefig(os.path.join(dirname, 'mel_dec_outputs'))
        fig = plot_spectrogram(output_mel.T)
        fig.savefig(os.path.join(dirname, 'mel_postnet_outputs'))

    @torch.no_grad()
    def eval_step(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        inputs_emotion = batch['input_emotions'].to(self.device)
        inputs_speaker = batch['input_speakers'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        valid_output_lengths = batch['valid_output_lengths'].to(self.device)
        mel_targets = batch['mel_targets'].to(self.device)
        durations = (
            batch['durations'].to(self.device)
            if batch['durations'] is not None else None)
        pitch_contours = batch['pitch_contours'].to(self.device)
        energy_contours = batch['energy_contours'].to(self.device)
        attn_priors = (
            batch['attn_priors'].to(self.device)
            if batch['attn_priors'] is not None else None)
        fp_label = None
        if self.fp_enable:
            fp_label = batch['fp_label'].to(self.device)
        # generate mel spectrograms
        res = self.model['KanTtsSAMBERT'](
            inputs_ling,
            inputs_emotion,
            inputs_speaker,
            valid_input_lengths,
            output_lengths=valid_output_lengths,
            mel_targets=mel_targets,
            duration_targets=durations,
            pitch_targets=pitch_contours,
            energy_targets=energy_contours,
            attn_priors=attn_priors,
            fp_label=fp_label,
        )

        x_band_width = res['x_band_width']
        h_band_width = res['h_band_width']
        dec_outputs = res['dec_outputs']
        postnet_outputs = res['postnet_outputs']
        log_duration_predictions = res['log_duration_predictions']
        pitch_predictions = res['pitch_predictions']
        energy_predictions = res['energy_predictions']
        duration_targets = res['duration_targets']
        pitch_targets = res['pitch_targets']
        energy_targets = res['energy_targets']
        fp_predictions = res['fp_predictions']
        valid_inter_lengths = res['valid_inter_lengths']

        mel_loss_, mel_loss = self.criterion['MelReconLoss'](
            valid_output_lengths, mel_targets, dec_outputs, postnet_outputs)

        dur_loss, pitch_loss, energy_loss = self.criterion['ProsodyReconLoss'](
            valid_inter_lengths,
            duration_targets,
            pitch_targets,
            energy_targets,
            log_duration_predictions,
            pitch_predictions,
            energy_predictions,
        )
        loss_total = mel_loss_ + mel_loss + dur_loss + pitch_loss + energy_loss
        if self.fp_enable:
            fp_loss = self.criterion['FpCELoss'](valid_input_lengths,
                                                 fp_predictions, fp_label)
            loss_total = loss_total + fp_loss

        if self.with_MAS:
            attn_soft = res['attn_soft']
            attn_hard = res['attn_hard']
            attn_logprob = res['attn_logprob']
            attn_ctc_loss = self.criterion['AttentionCTCLoss'](
                attn_logprob, valid_input_lengths, valid_output_lengths)
            attn_kl_loss = self.criterion['AttentionBinarizationLoss'](
                self.epoch, attn_hard, attn_soft)

            loss_total += attn_ctc_loss + attn_kl_loss
            self.total_eval_loss['eval/attn_ctc_loss'] += attn_ctc_loss.item()
            self.total_eval_loss['eval/attn_kl_loss'] += attn_kl_loss.item()

        self.total_eval_loss['eval/TotalLoss'] += loss_total.item()
        self.total_eval_loss['eval/mel_loss_'] += mel_loss_.item()
        self.total_eval_loss['eval/mel_loss'] += mel_loss.item()
        self.total_eval_loss['eval/dur_loss'] += dur_loss.item()
        self.total_eval_loss['eval/pitch_loss'] += pitch_loss.item()
        self.total_eval_loss['eval/energy_loss'] += energy_loss.item()
        if self.fp_enable:
            self.total_eval_loss['eval/fp_loss'] += fp_loss.item()
        self.total_eval_loss['eval/batch_size'] += mel_targets.size(0)
        self.total_eval_loss['eval/x_band_width'] += x_band_width
        self.total_eval_loss['eval/h_band_width'] += h_band_width

    def train_step(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        inputs_emotion = batch['input_emotions'].to(self.device)
        inputs_speaker = batch['input_speakers'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        valid_output_lengths = batch['valid_output_lengths'].to(self.device)
        mel_targets = batch['mel_targets'].to(self.device)
        durations = (
            batch['durations'].to(self.device)
            if batch['durations'] is not None else None)
        pitch_contours = batch['pitch_contours'].to(self.device)
        energy_contours = batch['energy_contours'].to(self.device)
        attn_priors = (
            batch['attn_priors'].to(self.device)
            if batch['attn_priors'] is not None else None)
        fp_label = None
        if self.fp_enable:
            fp_label = batch['fp_label'].to(self.device)

        # generate mel spectrograms
        res = self.model['KanTtsSAMBERT'](
            inputs_ling,
            inputs_emotion,
            inputs_speaker,
            valid_input_lengths,
            output_lengths=valid_output_lengths,
            mel_targets=mel_targets,
            duration_targets=durations,
            pitch_targets=pitch_contours,
            energy_targets=energy_contours,
            attn_priors=attn_priors,
            fp_label=fp_label,
        )

        x_band_width = res['x_band_width']
        h_band_width = res['h_band_width']
        dec_outputs = res['dec_outputs']
        postnet_outputs = res['postnet_outputs']
        log_duration_predictions = res['log_duration_predictions']
        pitch_predictions = res['pitch_predictions']
        energy_predictions = res['energy_predictions']

        duration_targets = res['duration_targets']
        pitch_targets = res['pitch_targets']
        energy_targets = res['energy_targets']
        fp_predictions = res['fp_predictions']
        valid_inter_lengths = res['valid_inter_lengths']

        mel_loss_, mel_loss = self.criterion['MelReconLoss'](
            valid_output_lengths, mel_targets, dec_outputs, postnet_outputs)

        dur_loss, pitch_loss, energy_loss = self.criterion['ProsodyReconLoss'](
            valid_inter_lengths,
            duration_targets,
            pitch_targets,
            energy_targets,
            log_duration_predictions,
            pitch_predictions,
            energy_predictions,
        )
        loss_total = mel_loss_ + mel_loss + dur_loss + pitch_loss + energy_loss
        if self.fp_enable:
            fp_loss = self.criterion['FpCELoss'](valid_input_lengths,
                                                 fp_predictions, fp_label)
            loss_total = loss_total + fp_loss

        if self.with_MAS:
            attn_soft = res['attn_soft']
            attn_hard = res['attn_hard']
            attn_logprob = res['attn_logprob']
            attn_ctc_loss = self.criterion['AttentionCTCLoss'](
                attn_logprob, valid_input_lengths, valid_output_lengths)
            attn_kl_loss = self.criterion['AttentionBinarizationLoss'](
                self.epoch, attn_hard, attn_soft)

            loss_total += attn_ctc_loss + attn_kl_loss
            self.total_train_loss['train/attn_ctc_loss'] += attn_ctc_loss.item(
            )
            self.total_train_loss['train/attn_kl_loss'] += attn_kl_loss.item()

        self.total_train_loss['train/TotalLoss'] += loss_total.item()
        self.total_train_loss['train/mel_loss_'] += mel_loss_.item()
        self.total_train_loss['train/mel_loss'] += mel_loss.item()
        self.total_train_loss['train/dur_loss'] += dur_loss.item()
        self.total_train_loss['train/pitch_loss'] += pitch_loss.item()
        self.total_train_loss['train/energy_loss'] += energy_loss.item()
        if self.fp_enable:
            self.total_train_loss['train/fp_loss'] += fp_loss.item()
        self.total_train_loss['train/batch_size'] += mel_targets.size(0)
        self.total_train_loss['train/x_band_width'] += x_band_width
        self.total_train_loss['train/h_band_width'] += h_band_width

        self.optimizer['KanTtsSAMBERT'].zero_grad()
        loss_total.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model['KanTtsSAMBERT'].parameters(), self.grad_clip)
        self.optimizer['KanTtsSAMBERT'].step()
        self.scheduler['KanTtsSAMBERT'].step()

    def save_checkpoint(self, checkpoint_path):
        if not self.distributed:
            model_state = self.model['KanTtsSAMBERT'].state_dict()
        else:
            model_state = self.model['KanTtsSAMBERT'].module.state_dict()
        state_dict = {
            'optimizer': self.optimizer['KanTtsSAMBERT'].state_dict(),
            'scheduler': self.scheduler['KanTtsSAMBERT'].state_dict(),
            'steps': self.steps,
            'model': model_state,
        }

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self,
                        checkpoint_path,
                        restore_training_state=False,
                        strict=True):
        state_dict = torch.load(checkpoint_path)
        if not self.distributed:
            self.model['KanTtsSAMBERT'].load_state_dict(
                state_dict['model'], strict=strict)
        else:
            self.model['KanTtsSAMBERT'].module.load_state_dict(
                state_dict['model'], strict=strict)

        if restore_training_state:
            self.optimizer['KanTtsSAMBERT'].load_state_dict(
                state_dict['optimizer'])
            self.scheduler['KanTtsSAMBERT'].load_state_dict(
                state_dict['scheduler'])
            self.steps = state_dict['steps']


class Textsy_BERT_Trainer(Trainer):

    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        sampler,
        train_loader,
        valid_loader,
        max_epochs=None,
        max_steps=None,
        save_dir=None,
        save_interval=1,
        valid_interval=1,
        log_interval=10,
        grad_clip=None,
    ):
        super().__init__(
            config,
            model,
            optimizer,
            scheduler,
            criterion,
            device,
            sampler,
            train_loader,
            valid_loader,
            max_epochs,
            max_steps,
            save_dir,
            save_interval,
            valid_interval,
            log_interval,
            grad_clip,
        )

    @torch.no_grad()
    def genearete_and_save_intermediate_result(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        bert_masks = batch['bert_masks'].to(self.device)
        targets = batch['targets'].to(self.device)

        res = self.model['KanTtsTextsyBERT'](
            inputs_ling[0:1],
            valid_input_lengths[0:1],
        )

        logits = res['logits']
        enc_slf_attn_lst = res['enc_slf_attn_lst']
        preds = torch.argmax(logits, dim=-1).contiguous().view(-1)

        dirname = os.path.join(self.log_dir, f'predictions/{self.steps}steps')
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for layer_id, slf_attn in enumerate(enc_slf_attn_lst):
            for head_id in range(self.config['Model']['KanTtsTextsyBERT']
                                 ['params']['encoder_num_heads']):
                fig = plot_alignment(
                    slf_attn[head_id, :valid_input_lengths[0], :
                             valid_input_lengths[0]].cpu().numpy(),
                    info='valid_len_{}'.format(valid_input_lengths[0].item()),
                )
                fig.savefig(
                    os.path.join(
                        dirname,
                        'enc_slf_attn_dev_layer{}_head{}'.format(
                            layer_id, head_id),
                    ))

        target = targets[0].cpu().numpy()
        bert_mask = bert_masks[0].cpu().numpy()
        pred = preds.cpu().numpy()
        np.save(os.path.join(dirname, 'pred.npy'), pred)
        np.save(os.path.join(dirname, 'target.npy'), target)
        np.save(os.path.join(dirname, 'bert_mask.npy'), bert_mask)

    @torch.no_grad()
    def eval_step(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        bert_masks = batch['bert_masks'].to(self.device)
        targets = batch['targets'].to(self.device)

        res = self.model['KanTtsTextsyBERT'](
            inputs_ling,
            valid_input_lengths,
        )

        logits = res['logits']
        loss_total, err = self.criterion['SeqCELoss'](
            logits,
            targets,
            bert_masks,
        )
        loss_total = loss_total / logits.size(-1)

        self.total_eval_loss['eval/TotalLoss'] += loss_total.item()
        self.total_eval_loss['eval/Error'] += err.item()
        self.total_eval_loss['eval/batch_size'] += targets.size(0)

    def train_step(self, batch):
        inputs_ling = batch['input_lings'].to(self.device)
        valid_input_lengths = batch['valid_input_lengths'].to(self.device)
        bert_masks = batch['bert_masks'].to(self.device)
        targets = batch['targets'].to(self.device)

        res = self.model['KanTtsTextsyBERT'](
            inputs_ling,
            valid_input_lengths,
        )

        logits = res['logits']
        loss_total, err = self.criterion['SeqCELoss'](
            logits,
            targets,
            bert_masks,
        )
        loss_total = loss_total / logits.size(-1)

        self.optimizer['KanTtsTextsyBERT'].zero_grad()
        loss_total.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model['KanTtsTextsyBERT'].parameters(), self.grad_clip)
        self.optimizer['KanTtsTextsyBERT'].step()
        self.scheduler['KanTtsTextsyBERT'].step()

        self.total_train_loss['train/TotalLoss'] += loss_total.item()
        self.total_train_loss['train/Error'] += err.item()
        self.total_train_loss['train/batch_size'] += targets.size(0)

    def save_checkpoint(self, checkpoint_path):
        if not self.distributed:
            model_state = self.model['KanTtsTextsyBERT'].state_dict()
        else:
            model_state = self.model['KanTtsTextsyBERT'].module.state_dict()
        state_dict = {
            'optimizer': self.optimizer['KanTtsTextsyBERT'].state_dict(),
            'scheduler': self.scheduler['KanTtsTextsyBERT'].state_dict(),
            'steps': self.steps,
            'model': model_state,
        }

        if not os.path.exists(checkpoint_path):
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self,
                        checkpoint_path,
                        restore_training_state=False,
                        strict=True):
        state_dict = torch.load(checkpoint_path)
        if not self.distributed:
            self.model['KanTtsTextsyBERT'].load_state_dict(
                state_dict['model'], strict=strict)
        else:
            self.model['KanTtsTextsyBERT'].module.load_state_dict(
                state_dict['model'], strict=strict)

        if restore_training_state:
            self.optimizer['KanTtsTextsyBERT'].load_state_dict(
                state_dict['optimizer'])
            self.scheduler['KanTtsTextsyBERT'].load_state_dict(
                state_dict['scheduler'])
            self.steps = state_dict['steps']
