# Copyright (c) 2019 NVIDIA Corporation
import argparse
import copy
import os
from functools import partial

from ruamel.yaml import YAML

import nemo
import nemo.collections.asr as nemo_asr
import nemo.utils.argparse as nm_argparse
from nemo.collections.asr.helpers import monitor_asr_train_progress, process_evaluation_batch, process_evaluation_epoch
from nemo.utils import logging
from nemo.utils.lr_policies import CosineAnnealing

import torch
#torch.autograd.set_detect_anomaly(True)


def parse_args():
    parser = argparse.ArgumentParser(
        parents=[nm_argparse.NemoArgParser()], description='Conformer', conflict_handler='resolve',
    )
    parser.set_defaults(
        checkpoint_dir=None,
        optimizer="adam",
        batch_size=32,
        eval_batch_size=32,
        lr=0.05 / (144 ** (0.5)),
        weight_decay=0.001,
        amp_opt_level="O0",
        create_tb_writer=True,
    )

    # Overwrite default args
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        required=True,
        help="number of epochs to train. You should specify either num_epochs or max_steps",
    )
    parser.add_argument(
        "--model_config", type=str, required=True, help="model configuration file: model.yaml",
    )

    # Create new args
    parser.add_argument("--exp_name", default="Conformer", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.98, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=int)
    parser.add_argument("--load_dir", default=None, type=str)
    parser.add_argument("--synced_bn", action='store_true', help="Use synchronized batch norm")
    parser.add_argument("--synced_bn_groupsize", default=0, type=int)

    parser.add_argument("--grad_norm_clip", type=float, default=-1, help="gradient clipping")
    parser.add_argument("--optimizer_eps", type=float, default=1e-9, help="The eps param of the optimizers.")

    parser.add_argument("--lstm_hidden_size", default=0, type=int)
    parser.add_argument("--wandb_project", default="Conformer", type=str)
    parser.add_argument("--wandb_exp", default=None, type=str)

    args = parser.parse_args()
    if args.max_steps is not None:
        raise ValueError("QuartzNet uses num_epochs instead of max_steps")

    return args


def construct_name(name, lr, batch_size, num_epochs, wd, optimizer):
    return "{0}-lr_{1}-bs_{2}-e_{3}-wd_{4}-opt_{5}".format(name, lr, batch_size, num_epochs, wd, optimizer)


def create_all_dags(args, neural_factory):
    '''
    creates train and eval dags as well as their callbacks
    returns train loss tensor and callbacks'''

    # parse the config files
    yaml = YAML(typ="safe")
    with open(args.model_config) as f:
        conformer_params = yaml.load(f)

    vocab = conformer_params['labels']
    sample_rate = conformer_params['sample_rate']

    # Calculate num_workers for dataloader
    total_cpus = os.cpu_count()
    cpu_per_traindl = max(int(total_cpus / neural_factory.world_size), 1)

    # create data layer for training
    train_dl_params = copy.deepcopy(conformer_params["AudioToTextDataLayer"])
    train_dl_params.update(conformer_params["AudioToTextDataLayer"]["train"])
    del train_dl_params["train"]
    del train_dl_params["eval"]
    # del train_dl_params["normalize_transcripts"]

    data_layer_train = nemo_asr.AudioToTextDataLayer(
        manifest_filepath=args.train_dataset,
        sample_rate=sample_rate,
        labels=vocab,
        batch_size=args.batch_size,
        num_workers=cpu_per_traindl,
        **train_dl_params,
        # normalize_transcripts=False
    )

    N = len(data_layer_train)
    steps_per_epoch = int(N / (args.batch_size * args.iter_per_step * args.num_gpus))

    # create separate data layers for eval
    # we need separate eval dags for separate eval datasets
    # but all other modules in these dags will be shared

    eval_dl_params = copy.deepcopy(conformer_params["AudioToTextDataLayer"])
    eval_dl_params.update(conformer_params["AudioToTextDataLayer"]["eval"])
    del eval_dl_params["train"]
    del eval_dl_params["eval"]

    data_layers_eval = []
    if args.eval_datasets:
        for eval_dataset in args.eval_datasets:
            data_layer_eval = nemo_asr.AudioToTextDataLayer(
                manifest_filepath=eval_dataset,
                sample_rate=sample_rate,
                labels=vocab,
                batch_size=args.eval_batch_size,
                num_workers=cpu_per_traindl,
                **eval_dl_params,
            )

            data_layers_eval.append(data_layer_eval)
    else:
        logging.warning("There were no val datasets passed")

    # create shared modules

    data_preprocessor = nemo_asr.AudioToMelSpectrogramPreprocessor(
        sample_rate=sample_rate, **conformer_params["AudioToMelSpectrogramPreprocessor"],
    )

    encoder = nemo_asr.ConformerEncoder(
        feat_in=conformer_params["AudioToMelSpectrogramPreprocessor"]["features"],
        lstm_hidden_size=args.lstm_hidden_size,
        **conformer_params["ConformerEncoder"],
    )

    decoder = nemo_asr.JasperDecoderForCTC(
        feat_in=conformer_params["ConformerEncoder"]["d_model"], num_classes=len(vocab),
    )

    ctc_loss = nemo_asr.CTCLossNM(num_classes=len(vocab), zero_infinity=True)

    greedy_decoder = nemo_asr.GreedyCTCDecoder()

    # create augmentation modules (only used for training) if their configs
    # are present

    multiply_batch_config = conformer_params.get('MultiplyBatch', None)
    if multiply_batch_config:
        multiply_batch = nemo_asr.MultiplyBatch(**multiply_batch_config)

    spectr_augment_config = conformer_params.get('SpectrogramAugmentation', None)
    if spectr_augment_config:
        data_spectr_augmentation = nemo_asr.SpectrogramAugmentation(**spectr_augment_config)

    # assemble train DAG

    (audio_signal_t, a_sig_length_t, transcript_t, transcript_len_t,) = data_layer_train()

    processed_signal_t, p_length_t = data_preprocessor(input_signal=audio_signal_t, length=a_sig_length_t)

    if multiply_batch_config:
        (processed_signal_t, p_length_t, transcript_t, transcript_len_t,) = multiply_batch(
            in_x=processed_signal_t, in_x_len=p_length_t, in_y=transcript_t, in_y_len=transcript_len_t,
        )

    if spectr_augment_config:
        processed_signal_t = data_spectr_augmentation(input_spec=processed_signal_t)

    encoded_t, encoded_len_t = encoder(audio_signal=processed_signal_t, length=p_length_t)
    log_probs_t = decoder(encoder_output=encoded_t)
    predictions_t = greedy_decoder(log_probs=log_probs_t)
    loss_t = ctc_loss(
        log_probs=log_probs_t, targets=transcript_t, input_length=encoded_len_t, target_length=transcript_len_t,
    )

    # create train callbacks
    train_callback = nemo.core.SimpleLossLoggerCallback(
        tensors=[loss_t, predictions_t, transcript_t, transcript_len_t],
        print_func=partial(monitor_asr_train_progress, labels=vocab),
        get_tb_values=lambda x: [["loss", x[0]]],
        tb_writer=neural_factory.tb_writer,
    )

    callbacks = [train_callback]

    if args.checkpoint_dir or args.load_dir:
        chpt_callback = nemo.core.CheckpointCallback(
            folder=args.checkpoint_dir, load_from_folder=args.load_dir, step_freq=args.checkpoint_save_freq,
        )

        callbacks.append(chpt_callback)

    if args.wandb_exp is not None:
        wand_callback = nemo.core.WandbCallback(
            train_tensors=[loss_t],
            wandb_name=args.wandb_exp,
            wandb_project=args.wandb_project,
            update_freq=1,  # args.loss_log_freq if args.loss_log_freq > 0 else steps_per_epoch,
            args=args,
        )
        callbacks.append(wand_callback)

    # assemble eval DAGs
    for i, eval_dl in enumerate(data_layers_eval):
        (audio_signal_e, a_sig_length_e, transcript_e, transcript_len_e,) = eval_dl()
        processed_signal_e, p_length_e = data_preprocessor(input_signal=audio_signal_e, length=a_sig_length_e)
        encoded_e, encoded_len_e = encoder(audio_signal=processed_signal_e, length=p_length_e)
        log_probs_e = decoder(encoder_output=encoded_e)
        predictions_e = greedy_decoder(log_probs=log_probs_e)
        loss_e = ctc_loss(
            log_probs=log_probs_e, targets=transcript_e, input_length=encoded_len_e, target_length=transcript_len_e,
        )

        # create corresponding eval callback
        tagname = os.path.basename(args.eval_datasets[i]).split(".")[0]

        eval_callback = nemo.core.EvaluatorCallback(
            eval_tensors=[loss_e, predictions_e, transcript_e, transcript_len_e,],
            user_iter_callback=partial(process_evaluation_batch, labels=vocab),
            user_epochs_done_callback=partial(process_evaluation_epoch, tag=tagname),
            eval_step=args.eval_freq,
            tb_writer=neural_factory.tb_writer,
            wandb_name=args.wandb_exp,
            wandb_project=args.wandb_project,
        )

        callbacks.append(eval_callback)

    return loss_t, callbacks, steps_per_epoch


def main():
    args = parse_args()

    name = construct_name(args.exp_name, args.lr, args.batch_size, args.num_epochs, args.weight_decay, args.optimizer,)
    work_dir = name
    if args.work_dir:
        work_dir = os.path.join(args.work_dir, name)

    # instantiate Neural Factory with supported backend
    neural_factory = nemo.core.NeuralModuleFactory(
        local_rank=args.local_rank,
        optimization_level=args.amp_opt_level,
        log_dir=work_dir,
        checkpoint_dir=args.checkpoint_dir,
        create_tb_writer=args.create_tb_writer,
        files_to_copy=[args.model_config, __file__],
        cudnn_benchmark=args.cudnn_benchmark,
        tensorboard_dir=args.tensorboard_dir,
        add_time_to_log_dir=True,
    )
    args.num_gpus = neural_factory.world_size

    args.checkpoint_dir = neural_factory.checkpoint_dir

    if args.local_rank is not None:
        logging.info('Doing ALL GPU')

    # build dags
    train_loss, callbacks, steps_per_epoch = create_all_dags(args, neural_factory)

    grad_norm_clip = args.grad_norm_clip if args.grad_norm_clip > 0 else None

    # train model
    neural_factory.train(
        tensors_to_optimize=[train_loss],
        callbacks=callbacks,
        lr_policy=CosineAnnealing(args.num_epochs * steps_per_epoch, warmup_steps=args.warmup_steps),
        optimizer=args.optimizer,
        optimization_params={
            "num_epochs": args.num_epochs,
            "lr": args.lr,
            "betas": (args.beta1, args.beta2),
            "weight_decay": args.weight_decay,
            "grad_norm_clip": grad_norm_clip,
            "eps": args.optimizer_eps,
        },
        batches_per_step=args.iter_per_step,
        synced_batchnorm=args.synced_bn,
        synced_batchnorm_groupsize=args.synced_bn_groupsize,
    )


if __name__ == '__main__':
    main()
