# Copyright (c) 2019 NVIDIA Corporation
from functools import partial

import torch

import nemo
from nemo.collections.asr.helpers import __gather_losses
from nemo.collections.asr.metrics import classification_accuracy, word_error_rate
from nemo.collections.asr.parts.beam_search_rnnt import rnnt_beam_decode_dynamic, rnnt_beam_decode_static
from nemo.collections.nlp.data.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.utils import logging

logging = nemo.logging

ALLOWED_RNNT_DECODING_SCHEMES = ['static', 'dynamic']


def __ctc_decoder_predictions_tensor(tensor, tokenizer: TokenizerSpec):
    """
    Decodes a sequence of labels to words
    """
    blank_id = tokenizer.tokenizer.vocab_size
    hypotheses = []
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = blank_id  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = tokenizer.ids_to_text(decoded_prediction)
        hypotheses.append(hypothesis)
    return hypotheses


def __rnnt_decoder_predictions_list(decoded_predictions, tokenizer: TokenizerSpec, decoder_type, beam_size=1):
    """
    Decodes a sequence of labels to words
    """
    blank_id = tokenizer.tokenizer.vocab_size
    hypotheses = []
    # iterate over batch
    decoded_predictions = decoded_predictions.cpu().numpy()

    if decoder_type == 'static':
        decoded_prediction = rnnt_beam_decode_static(decoded_predictions, blank_id, beam_size=beam_size,
                                                     parallel=True)
    elif decoder_type == 'dynamic':
        decoded_prediction = rnnt_beam_decode_dynamic(decoded_predictions, blank_id, parallel=True)
    else:
        raise ValueError('`decoder_type` can only be one of {}'.format(str(ALLOWED_RNNT_DECODING_SCHEMES)))

    for ind in range(decoded_predictions.shape[0]):
        hypothesis = tokenizer.ids_to_text(decoded_prediction[ind])
        hypotheses.append(hypothesis)

    return hypotheses


def monitor_asr_train_progress(tensors: list, tokenizer: TokenizerSpec, eval_metric='WER', tb_logger=None):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints sample to screen, computes
    and logs AVG WER to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (predictions, targets, target_lengths)
      labels: A list of labels
      eval_metric: An optional string from 'WER', 'CER'. Defaults to 'WER'.
      tb_logger: Tensorboard logging object
    Returns:
      None
    """
    references = []

    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[2].long().cpu()
        tgt_lenths_cpu_tensor = tensors[3].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = tokenizer.ids_to_text(target)
            references.append(reference)

        hypotheses = __ctc_decoder_predictions_tensor(tensors[1], tokenizer=tokenizer)

    eval_metric = eval_metric.upper()
    if eval_metric not in {'WER', 'CER'}:
        raise ValueError('eval_metric must be \'WER\' or \'CER\'')
    use_cer = True if eval_metric == 'CER' else False

    tag = f'training_batch_{eval_metric}'
    wer = word_error_rate(hypotheses, references, use_cer=use_cer)
    if tb_logger is not None:
        tb_logger.add_scalar(tag, wer)
    logging.info(f'Loss: {tensors[0]}')
    logging.info(f'{tag}: {wer * 100 : 5.2f}%')
    logging.info(f'Prediction: {hypotheses[0]}')
    logging.info(f'Reference: {references[0]}')


def monitor_transducer_asr_train_progress(
    tensors: list,
    tokenizer: TokenizerSpec,
    eval_metric: str = 'WER',
    tb_logger=None,
    beam_size: int = 1,
    decoder_type: str = 'static',
):
    """
    Takes output of greedy ctc decoder and performs ctc decoding algorithm to
    remove duplicates and special symbol. Prints sample to screen, computes
    and logs AVG WER to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (predictions, targets, target_lengths)
      labels: A list of labels
      eval_metric: An optional string from 'WER', 'CER'. Defaults to 'WER'.
      tb_logger: Tensorboard logging object
      beam_size: Integer value to select beam size for static or dynamic beam
        search. Only used for RNNT decoding. Must be greater than or equal to 1.
      decode_type: String value to select whether to perform static decoding or
        or dynamic decoding. Static decoding is faster, but gives overly optimistic
        results. Dynamic decoding should be used for evaluating at test time.
    Returns:
      None
    """

    if beam_size < 1:
        raise ValueError('`beam_size` must be greater >= 1')

    references = []

    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[2].long().cpu()
        tgt_lenths_cpu_tensor = tensors[3].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = tokenizer.ids_to_text(target)
            references.append(reference)

        hypotheses = __rnnt_decoder_predictions_list(
            tensors[1], tokenizer=tokenizer, decoder_type=decoder_type, beam_size=beam_size
        )

    eval_metric = eval_metric.upper()
    if eval_metric not in {'WER', 'CER'}:
        raise ValueError('eval_metric must be \'WER\' or \'CER\'')
    use_cer = True if eval_metric == 'CER' else False

    tag = f'training_batch_{eval_metric}'
    wer = word_error_rate(hypotheses, references, use_cer=use_cer)
    if tb_logger is not None:
        tb_logger.add_scalar(tag, wer)
    logging.info(f'Loss: {tensors[0]}')
    logging.info(f'{tag}: {wer * 100 : 5.2f}%')
    logging.info(f'Prediction: {hypotheses[0]}')
    logging.info(f'Reference: {references[0]}')


def __gather_predictions(predictions_list: list, tokenizer: TokenizerSpec, decoder_type: str, beam_size: int) -> list:
    results = []
    if decoder_type == 'ctc':
        decoder_func = __ctc_decoder_predictions_tensor
    else:
        decoder_func = partial(__rnnt_decoder_predictions_list, decoder_type=decoder_type, beam_size=beam_size,
                               parallel=True)

    for prediction in predictions_list:
        results += decoder_func(prediction, tokenizer=tokenizer)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list, tokenizer: TokenizerSpec) -> list:
    results = []
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = tokenizer.ids_to_text(target)
            results.append(reference)
    return results


def process_evaluation_batch(tensors: dict, global_vars: dict, tokenizer: TokenizerSpec):
    """
    Creates a dictionary holding the results from a batch of audio
    """
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'predictions' not in global_vars.keys():
        global_vars['predictions'] = []
    if 'transcripts' not in global_vars.keys():
        global_vars['transcripts'] = []
    if 'logits' not in global_vars.keys():
        global_vars['logits'] = []
    # if not 'transcript_lengths' in global_vars.keys():
    #  global_vars['transcript_lengths'] = []
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(v, tokenizer=tokenizer, decoder_type='ctc', beam_size=0)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list, transcript_len_list, tokenizer=tokenizer)


def process_transducer_evaluation_batch(
    tensors: dict, global_vars: dict, tokenizer: TokenizerSpec, decoder_type: str = 'ctc', beam_size: int = 1
):
    """
    Creates a dictionary holding the results from a batch of audio
    """
    if decoder_type not in ALLOWED_RNNT_DECODING_SCHEMES:
        raise ValueError('`decoder_type` must be either {}'.format(str(*ALLOWED_RNNT_DECODING_SCHEMES)))

    if beam_size < 1:
        raise ValueError('`beam_size` must be greater >= 1')

    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'predictions' not in global_vars.keys():
        global_vars['predictions'] = []
    if 'transcripts' not in global_vars.keys():
        global_vars['transcripts'] = []
    if 'logits' not in global_vars.keys():
        global_vars['logits'] = []
    # if not 'transcript_lengths' in global_vars.keys():
    #  global_vars['transcript_lengths'] = []
    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('predictions'):
            global_vars['predictions'] += __gather_predictions(
                v, tokenizer=tokenizer, decoder_type=decoder_type, beam_size=beam_size
            )
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list, transcript_len_list, tokenizer=tokenizer)


def post_process_predictions(predictions, tokenizer: TokenizerSpec, decoder_type: str = 'ctc', beam_size: int = 1):
    if decoder_type not in ('ctc', *ALLOWED_RNNT_DECODING_SCHEMES):
        raise ValueError('`decoder_type` must be either `ctc` or {}'.format(str(ALLOWED_RNNT_DECODING_SCHEMES)))

    return __gather_predictions(predictions, tokenizer=tokenizer, decoder_type=decoder_type, beam_size=beam_size)


def post_process_transcripts(transcript_list, transcript_len_list, tokenizer: TokenizerSpec):
    return __gather_transcripts(transcript_list, transcript_len_list, tokenizer=tokenizer)
