"""Finetune utilities."""

import os
import pathlib
import random

import json
import mpu
import pretrain_glm
import torch
import torch.utils.data
from arguments import get_args
from configure_data import make_data_loader, prepare_tokenizer
from filelock import FileLock
from pretrain_glm import evaluate_and_print_results
from pretrain_glm import forward_step as lm_forward_step
from pretrain_glm import (initialize_distributed, report_iteration_metrics,
                          set_random_seed)
from tasks.data_utils import FakeDataloader, build_data_loader
from train_utils import load_pretrained, setup_model_and_optimizer, train_step
from utils import (Timers, debug_finetune_data, get_log_dir, get_sample_writer,
                   load_checkpoint, print_and_save_args, print_rank_0,
                   save_checkpoint)


def process_batch(batch, args):
    """Process batch and produce inputs for the model."""
    keys = ['text', 'label']
    if args.pretrained_bert:
        keys += ['padding_mask', 'types']
    else:
        keys += ['mask', 'position']
        if args.cloze_eval:
            if args.fast_decode:
                keys += [
                    'dec_text', 'dec_position', 'dec_mask', 'dec_target',
                    'dec_logit_mask'
                ]
            else:
                keys += ['target', 'logit_mask']
                if args.segment_length > 0:
                    keys += ['segment_id']
                if args.continuous_prompt:
                    keys += ['prompt_pos']
    if args.variable_num_choices:
        keys.append('loss_mask')
    # Broadcast data.
    datatype = torch.int64
    data_b = mpu.broadcast_data(keys, batch, datatype)

    if 'padding_mask' in data_b:
        attention_mask = data_b['padding_mask'].float().cuda().contiguous()
        if args.fp16:
            attention_mask = attention_mask.half()
        data_b['padding_mask'] = attention_mask
    return data_b


tokenizer = None


def mix_forward_step(batch_and_dataloader, model, args, times, mems):
    use_blocklm = 0
    if args.block_lm_ratio > 0.0:
        if mpu.get_model_parallel_rank() == 0:
            if random.random() > 1 / (1 + args.block_lm_ratio):
                use_blocklm = 1
        use_blocklm = torch.cuda.LongTensor([use_blocklm])
        torch.distributed.broadcast(
            use_blocklm,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group())
        use_blocklm = use_blocklm.item()
    if use_blocklm:
        return lm_forward_step((batch_and_dataloader[1], None), model, args,
                               times, mems)
    else:
        return finetune_forward_step(batch_and_dataloader[0], model, args,
                                     times, mems)


def finetune_forward_step(batch, model, args, timers, mems):
    """Simple forward step with cross-entropy loss."""
    # Get the batch.
    timers('batch generator').start()
    try:
        batch_ = next(batch)
    except BaseException:
        batch_ = batch

    data = process_batch(batch_, args)
    timers('batch generator').stop()

    # Forward model.
    if args.pretrained_bert:
        tokens, types, labels, attention_mask = data['text'], data[
            'types'], data['label'], data['padding_mask']
        logits = model(
            tokens,
            token_type_ids=types,
            attention_mask=attention_mask,
            checkpoint_activations=True)
    elif args.cloze_eval:
        tokens, labels, position_ids = data['text'], data['label'], data[
            'position']
        attention_mask = data['mask']

        if not args.fast_decode:
            target_ids, logit_mask = data['target'], data['logit_mask']
            if args.continuous_prompt:
                prompt_pos = data['prompt_pos']
                result = model(
                    tokens,
                    position_ids,
                    attention_mask,
                    target_ids,
                    logit_mask,
                    prompt_pos=prompt_pos)
            else:
                result = model(tokens, position_ids, attention_mask,
                               target_ids, logit_mask)
            if not args.multi_token:
                logits, lm_logits, *mems = result
            else:
                logits, *mems = result
        else:
            dec_input_ids, dec_position_ids, dec_attention_mask = data[
                'dec_text'], data['dec_position'], data['dec_mask']
            dec_target_ids, dec_logit_mask = data['dec_target'], data[
                'dec_logit_mask']
            logits, *mems = model(tokens, position_ids, attention_mask,
                                  dec_input_ids, dec_position_ids,
                                  dec_attention_mask, dec_target_ids,
                                  dec_logit_mask)
    else:
        tokens, labels, position_ids, attention_mask = data['text'], data[
            'label'], data['position'], data['mask']
        logits, *mems = model(tokens, position_ids, attention_mask)

    if args.adapet:
        batch_size, num_classes = logits.size()[:2]
        label_mask = torch.ones(batch_size, num_classes, device=logits.device)
        label_mask.scatter_(1, labels.unsqueeze(1), -1.0)
        if 'loss_mask' in data:
            loss_mask = data['loss_mask']
            label_mask = label_mask * loss_mask
        loss = logits.contiguous().float() * label_mask
        loss = loss.sum() / batch_size
    else:
        if 'segment_id' in data:
            from torch_scatter import scatter_sum
            if 'loss_mask' in data:
                logits = logits * data['loss_mask']
            logits = scatter_sum(logits, data['segment_id'], dim=1)
        elif 'loss_mask' in data:
            loss_mask = data['loss_mask']
            logits = logits * loss_mask - 10000.0 * (1.0 - loss_mask)
        if args.loss_func == 'cross_entropy':
            # Cross-entropy loss.
            loss_func = torch.nn.CrossEntropyLoss()
            loss = loss_func(logits.contiguous().float(), labels)
        elif args.loss_func == 'hinge':
            correct_logits = logits[range(logits.size(0)), labels]
            hinge_loss = 1 + logits - correct_logits.unsqueeze(1)
            hinge_loss[hinge_loss < 0.0] = 0.0
            loss = hinge_loss.sum(dim=1).mean() - 1.0
        elif args.loss_func == 'generative' or args.loss_func == 'mix':
            batch_size = logits.size(0)
            loss = -logits[range(batch_size), labels].mean()
            if args.loss_func == 'mix':
                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss + loss_func(logits.contiguous().float(), labels)
        else:
            raise NotImplementedError

    # Reduce loss for logging.

    return loss, mems, 'bert'


def _build_infinite_size_dataloader(dataloader):
    """Build a looped dataloader with infinite size."""

    iterator = dataloader.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = dataloader.__iter__()


def _build_train_valid_dataloaders(train_dataset, valid_dataset, args):
    """Traing and validation dataloaders."""
    print_rank_0('building train and validation dataloaders ...')
    # Training dataset.
    train_dataloader = build_data_loader(
        train_dataset, args.batch_size, args.num_workers, drop_last=False)
    # Set the training iterations.
    args.train_iters_per_epoch = len(train_dataloader)
    args.train_iters = args.epochs * args.train_iters_per_epoch
    # Validation dataset. For this dataset, we do not need to set up
    # shuffling so we can just use a simple infinite loop.
    valid_dataloader = None
    if valid_dataset is not None:
        valid_dataloader_ = build_data_loader(
            valid_dataset, args.batch_size, args.num_workers, drop_last=False)
        valid_dataloader = _build_infinite_size_dataloader(valid_dataloader_)

    return train_dataloader, valid_dataloader


def _train(model,
           optimizer,
           lr_scheduler,
           forward_step,
           train_dataloader,
           valid_dataloader,
           end_of_epoch_callback,
           args,
           timers,
           summary_writer=None):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    args.iteration = 0
    total_lm_loss = 0.0
    best_score, best_iteration = 0, None
    # Starting epoch and iteration
    start_epoch = args.iteration // args.train_iters_per_epoch
    start_iteration = args.iteration % args.train_iters_per_epoch
    if not args.block_lm_ratio:
        valid_dataloader = valid_dataloader[0]
    # For each remaining epoch
    timers('interval time').start()
    for epoch in range(start_epoch, args.epochs):
        print_rank_0('working on epoch {} ...'.format(epoch))

        # Set the data loader epoch to shuffle the index iterator.
        if mpu.get_model_parallel_rank() == 0:
            train_dataloader[0].sampler.set_epoch(args.seed + epoch)

        # For all the batches in the dataset.
        for iteration_, batch in enumerate(train_dataloader[0]):

            # Ignore the iterations before starting value
            if iteration_ < start_iteration:
                continue
            # Set to zero so the next epoch does not skip any batches.
            start_iteration = 0

            # Train for one step.
            if args.block_lm_ratio > 0.0:
                data = (batch, train_dataloader[1])
            else:
                data = batch
            lm_loss, skipped_iter, _ = train_step(
                data,
                model,
                optimizer,
                lr_scheduler,
                args,
                timers,
                forward_step_func=forward_step,
                single_step=True)
            args.iteration += 1
            total_lm_loss += lm_loss.data.detach().float()

            # Logging.
            if args.iteration % args.log_interval == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_lm_loss.item() / args.log_interval
                elapsed_time = timers('interval time').elapsed()
                timers.log([
                    'forward', 'backward', 'allreduce', 'optimizer',
                    'batch generator'
                ],
                           normalizer=args.log_interval)  # noqa
                report_iteration_metrics(
                    summary_writer, optimizer, learning_rate, avg_lm_loss,
                    elapsed_time * 1000.0 / args.log_interval, args.iteration,
                    args.train_iters, args)
                total_lm_loss = 0.0

            # Evaluation
            if args.eval_interval and valid_dataloader is not None and args.iteration % args.eval_interval == 0:
                prefix = 'iteration {}'.format(args.iteration)
                evaluate_and_print_results(
                    prefix,
                    valid_dataloader,
                    model,
                    args,
                    timers,
                    step=args.iteration,
                    verbose=False,
                    forward_step_func=forward_step,
                    summary_writer=summary_writer)

        # Checkpointing at the end of each epoch.
        if args.save and (epoch + 1) % args.save_epoch == 0:
            save_checkpoint(
                args.iteration,
                model,
                optimizer,
                lr_scheduler,
                args,
                only_changed_parameters=True)

        # Callback at the end of each epoch.
        if end_of_epoch_callback is not None and (epoch
                                                  + 1) % args.eval_epoch == 0:
            score_dict = end_of_epoch_callback(
                model, epoch, summary_writer=summary_writer)
            validation_metric = args.validation_metric if args.validation_metric else list(
                score_dict.keys())[0]
            validation_score = score_dict[validation_metric]
            if best_iteration is None or validation_score > best_score:
                best_iteration = args.iteration
                best_score = validation_score
                print_rank_0(
                    f'Found best {validation_metric} {best_score} at {best_iteration}'
                )
                save_checkpoint(
                    args.iteration,
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    tag='best',
                    barrier=False,
                    only_changed_parameters=True,
                    no_deepspeed=True,
                    no_save_optim=True)
                if torch.distributed.get_rank() == 0:
                    score_dict.update({'type': 'validation', 'epoch': epoch})
                    with open(os.path.join(args.log_dir, 'results.json'),
                              'w') as output:
                        output.write(json.dumps(score_dict) + '\n')
                    with open(
                            os.path.join(args.save,
                                         'best_checkpointed_iteration.txt'),
                            'w') as output:
                        output.write(str(best_iteration))
    torch.distributed.barrier()
    return best_iteration


def finetune(args,
             train_valid_datasets_provider,
             model_kwargs,
             forward_step=finetune_forward_step,
             end_of_epoch_callback_provider=None):
    """Main finetune function used across all tasks."""
    global tokenizer
    timers = Timers()
    tokenizer = prepare_tokenizer(args)
    pretrain_glm.tokenizer = tokenizer
    if args.save:
        args.save = os.path.join(args.save, args.experiment_name)
    # Train and validation data loaders.
    timers('train/valid/test dataset/dataloder').start()
    train_dataloader, valid_dataloader = None, None
    train_block_dataloader, valid_block_dataloader = None, None
    if train_valid_datasets_provider is not None and args.epochs > 0:
        if mpu.get_model_parallel_rank() == 0:
            train_dataset, valid_dataset = train_valid_datasets_provider(
                args, tokenizer)
            train_dataloader, valid_dataloader = _build_train_valid_dataloaders(
                train_dataset, valid_dataset, args)
            if args.no_validation:
                valid_dataloader = None
            train_iters = torch.cuda.LongTensor([len(train_dataloader)])
        else:
            train_iters = torch.cuda.LongTensor([0])
        torch.distributed.broadcast(
            train_iters,
            mpu.get_model_parallel_src_rank(),
            group=mpu.get_model_parallel_group())
        if mpu.get_model_parallel_rank() != 0:
            args.train_iters_per_epoch = train_iters[0].item()
            args.train_iters = args.epochs * args.train_iters_per_epoch

            train_dataloader = FakeDataloader(args.train_iters_per_epoch)
            if args.no_validation:
                valid_dataloader = None
            else:
                valid_dataloader = FakeDataloader(None)
        if args.block_lm_ratio > 0.0:
            if mpu.get_model_parallel_rank() == 0:
                train_block_dataset, valid_block_dataset = train_valid_datasets_provider(
                    args, tokenizer, pattern_text=True)
                train_block_dataloader = make_data_loader(
                    train_block_dataset,
                    tokenizer,
                    args.batch_size * mpu.get_data_parallel_world_size(),
                    args.train_iters,
                    args,
                    shuffle=True,
                    block_collate=True)
                valid_block_dataloader = make_data_loader(
                    valid_block_dataset,
                    tokenizer,
                    args.batch_size * mpu.get_data_parallel_world_size(),
                    (args.train_iters // args.eval_interval + 1)
                    * args.eval_iters,
                    args,
                    shuffle=True,
                    block_collate=True)
            else:
                train_block_dataloader = FakeDataloader(args.train_iters)
                valid_block_dataloader = FakeDataloader(None)
            train_block_dataloader, valid_block_dataloader = iter(
                train_block_dataloader), iter(valid_block_dataloader)

    timers('train/valid/test dataset/dataloder').stop()
    # Build calback function.
    timers('callback function').start()
    end_of_epoch_callback, end_of_train_callback = None, None
    if end_of_epoch_callback_provider is not None:
        if train_valid_datasets_provider is not None and args.epochs > 0 and not args.no_validation:
            end_of_epoch_callback = end_of_epoch_callback_provider(
                args, tokenizer, is_test=False)
        end_of_train_callback = end_of_epoch_callback_provider(
            args, tokenizer, is_test=True)
    timers('callback function').stop()

    # Build model, optimizer and learning rate scheduler.
    timers('model and optimizer').start()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(
        args, **model_kwargs)
    timers('model and optimizer').stop()

    # If pretrained checkpoint is provided and we have not trained for
    # any iteration (i.e., iteration is zero), then load the pretrained
    # checkpoint.
    timers('pretrained checkpoint').start()
    if args.load_pretrained is not None and not args.pretrained_bert:
        task_tokens = None
        if args.continuous_prompt and args.prompt_init:
            if mpu.get_model_parallel_rank() == 0:
                dataset = train_dataloader.dataset
                processor, pvp = dataset.processor, dataset.pvp
                task_tokens = []
                for label in processor.get_labels():
                    verbalizer = pvp.verbalize(label)[0]
                    verbalizer_ids = tokenizer.EncodeAsIds(
                        verbalizer).tokenization
                    task_tokens += verbalizer_ids
                print_rank_0('Task tokens: '
                             + tokenizer.DecodeIds(task_tokens))
                num_task_tokens = len(task_tokens)
            else:
                num_task_tokens, task_tokens = 0, []
            num_task_tokens = torch.cuda.LongTensor([num_task_tokens])
            torch.distributed.broadcast(
                num_task_tokens,
                mpu.get_model_parallel_src_rank(),
                group=mpu.get_model_parallel_group())
            num_task_tokens = num_task_tokens.item()
            if num_task_tokens > 0:
                if mpu.get_model_parallel_rank() == 0:
                    task_tokens = torch.cuda.LongTensor(task_tokens)
                else:
                    task_tokens = torch.empty(
                        num_task_tokens,
                        device=torch.cuda.current_device(),
                        dtype=torch.long)
                torch.distributed.broadcast(
                    task_tokens,
                    mpu.get_model_parallel_src_rank(),
                    group=mpu.get_model_parallel_group())
                task_tokens = task_tokens.tolist()
        with FileLock(
                os.path.join(pathlib.Path.home(), 'checkpoint_lock'),
                timeout=-1):
            load_pretrained(
                model, args.load_pretrained, args, task_tokens=task_tokens)
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16 and optimizer is not None:
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    if args.load is not None:
        with FileLock(
                os.path.join(pathlib.Path.home(), 'checkpoint_lock'),
                timeout=-1):
            load_checkpoint(
                model,
                optimizer,
                lr_scheduler,
                args,
                no_deepspeed=args.no_deepspeed_load)
        # This is critical when only model is loaded. We should make sure
        # master parameters are also updated.
        if args.fp16 and optimizer is not None:
            if args.deepspeed:
                optimizer.refresh_fp32_params()
            else:
                optimizer._model_params_to_master_params()
    torch.distributed.barrier()
    timers('pretrained checkpoint').stop()
    args.iteration = 0
    summary_writer = None
    if torch.distributed.get_rank() == 0:
        args.log_dir = get_log_dir(
            base=args.summary_dir, name=args.experiment_name)
        if os.path.exists(os.path.join(args.log_dir, 'test_results.json')
                          ) and args.load is None and not args.overwrite:
            raise ValueError(
                'Output directory ({}) already exists and is not empty.'.
                format(args.log_dir))
        summary_writer = get_sample_writer(
            log_dir=args.log_dir, iteration=args.iteration)
        print_and_save_args(args, verbose=True, log_dir=args.log_dir)

    # Print setup timing.
    print_rank_0('done with setups ...')
    timers.log([
        'train/valid/test dataset/dataloder', 'callback function',
        'model and optimizer', 'pretrained checkpoint'
    ])
    print_rank_0('training ...')

    # Finetune the model.
    score_dict = None
    if train_dataloader is not None and args.epochs > 0:
        if args.block_lm_ratio > 0.0:
            forward_step = mix_forward_step
        best_iteration = _train(
            model,
            optimizer,
            lr_scheduler,
            forward_step, (train_dataloader, train_block_dataloader),
            (valid_dataloader, valid_block_dataloader),
            end_of_epoch_callback,
            args,
            timers,
            summary_writer=summary_writer)
        if end_of_train_callback is not None and best_iteration is not None:
            with FileLock(
                    os.path.join(pathlib.Path.home(), 'checkpoint_lock'),
                    timeout=-1):
                args.load = os.path.join(args.save, 'best')
                load_checkpoint(
                    model,
                    optimizer,
                    lr_scheduler,
                    args,
                    no_load_optim=True,
                    no_deepspeed=True)
                args.load = None
        torch.distributed.barrier()
        if end_of_train_callback is not None:
            score_dict = end_of_train_callback(
                model, epoch=-1, output_predictions=True)
    # Or just evaluate.
    else:
        if end_of_train_callback is not None:
            print_rank_0('evaluation only mode, setting epoch to -1')
            score_dict = end_of_train_callback(
                model, epoch=-1, output_predictions=True)
    if score_dict is not None and torch.distributed.get_rank() == 0:
        score_dict.update({'type': 'test'})
        with open(os.path.join(args.log_dir, 'test_results.json'),
                  'w') as output:
            output.write(json.dumps(score_dict) + '\n')

    print_rank_0('done :-)')


if __name__ == '__main__':
    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()
    assert args.finetune

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    from tasks.superglue.dataset import PROCESSORS

    superglue_tasks = list(PROCESSORS.keys())
    if args.task.lower() in superglue_tasks:
        from tasks.superglue.finetune import main
    elif args.task.lower() in ['lambda', 'wikitext', 'language_model']:
        from tasks.language_model.finetune import main
    elif args.task.lower() in [
            'cnn_dm', 'cnn_dm_original', 'gigaword', 'blank',
            'squad_generation', 'xsum', 'extraction'
    ]:
        from tasks.seq2seq.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main(args)
