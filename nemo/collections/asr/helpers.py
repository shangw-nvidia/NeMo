# Copyright (c) 2019 NVIDIA Corporation
from functools import partial

import torch

import nemo
from nemo.collections.asr.metrics import classification_accuracy, word_error_rate
from nemo.collections.asr.parts.beam_search_rnnt import rnnt_beam_decode_dynamic, rnnt_beam_decode_static
from nemo.utils import logging

try:
    from nemo.collections.asr.parts import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False

logging = nemo.logging

ALLOWED_RNNT_DECODING_SCHEMES = ['static', 'dynamic']


def __ctc_decoder_predictions_tensor(tensor, labels):
    """
    Decodes a sequence of labels to words
    """
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    prediction_cpu_tensor = tensor.long().cpu()
    # iterate over batch
    for ind in range(prediction_cpu_tensor.shape[0]):
        prediction = prediction_cpu_tensor[ind].numpy().tolist()
        # CTC decoding procedure
        decoded_prediction = []
        previous = len(labels)  # id of a blank symbol
        for p in prediction:
            if (p != previous or previous == blank_id) and p != blank_id:
                decoded_prediction.append(p)
            previous = p
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction])
        hypotheses.append(hypothesis)
    return hypotheses


def __rnnt_decoder_predictions_list(decoded_predictions, labels, decoder_type, beam_size=1):
    """
    Decodes a sequence of labels to words
    """
    blank_id = len(labels)
    hypotheses = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over batch
    decoded_predictions = decoded_predictions.cpu().numpy()

    if decoder_type == 'static':
        decoded_prediction = rnnt_beam_decode_static(decoded_predictions, blank_id, beam_size=beam_size)
    elif decoder_type == 'dynamic':
        decoded_prediction = rnnt_beam_decode_dynamic(decoded_predictions, blank_id)
    else:
        raise ValueError('`decoder_type` can only be one of {}'.format(str(ALLOWED_RNNT_DECODING_SCHEMES)))

    for ind in range(decoded_predictions.shape[0]):
        hypothesis = ''.join([labels_map[c] for c in decoded_prediction[ind]])
        hypotheses.append(hypothesis)

    return hypotheses


def monitor_asr_train_progress(tensors: list, labels: list, eval_metric='WER', tb_logger=None):
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

    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[2].long().cpu()
        tgt_lenths_cpu_tensor = tensors[3].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            references.append(reference)

        hypotheses = __ctc_decoder_predictions_tensor(tensors[1], labels=labels)

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
    labels: list,
    eval_metric: str = 'WER',
    tb_logger=None,
    beam_size: int = 1,
    decode_type: str = 'static',
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
      decoder_type: String value to select type of decoding required.
        Can be `ctc` or `rnnt`.
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

    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    with torch.no_grad():
        # prediction_cpu_tensor = tensors[0].long().cpu()
        targets_cpu_tensor = tensors[2].long().cpu()
        tgt_lenths_cpu_tensor = tensors[3].long().cpu()

        # iterate over batch
        for ind in range(targets_cpu_tensor.shape[0]):
            tgt_len = tgt_lenths_cpu_tensor[ind].item()
            target = targets_cpu_tensor[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            references.append(reference)

        hypotheses = __rnnt_decoder_predictions_list(
            tensors[1], labels=labels, decoder_type=decode_type, beam_size=beam_size
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


def monitor_classification_training_progress(tensors: list, eval_metric=None, tb_logger=None):
    """
    Computes the top k classification accuracy of the model being trained.
    Prints sample to screen, computes and  and logs a list of top k accuracies
    to console and (optionally) Tensorboard
    Args:
      tensors: A list of 3 tensors (loss, logits, targets)
      eval_metric: An optional list of integers detailing Top@`k`
        in the range [1, max_classes]. Defaults to [1] if not set.
      tb_logger: Tensorboard logging object
    Returns:
      None
    """
    if eval_metric is None:
        eval_metric = [1]

    if type(eval_metric) not in (list, tuple):
        eval_metric = [eval_metric]

    top_k = eval_metric

    with torch.no_grad():
        logits, targets = tensors[1:]
        topk_acc = classification_accuracy(logits, targets, top_k=top_k)

    tag = 'training_batch_top@{0}'
    logging.info(f'Loss: {tensors[0]}')

    for k, acc in zip(top_k, topk_acc):
        if tb_logger is not None:
            tb_logger.add_scalar(tag.format(k), acc)

        logging.info(f"{tag.format(k)}: {acc * 100.: 3.4f}")


def __gather_losses(losses_list: list) -> list:
    return [torch.mean(torch.stack(losses_list))]


def __gather_predictions(predictions_list: list, labels: list, decoder_type: str, beam_size: int) -> list:
    results = []
    if decoder_type == 'ctc':
        decoder_func = __ctc_decoder_predictions_tensor
    else:
        decoder_func = partial(__rnnt_decoder_predictions_list, decoder_type=decoder_type, beam_size=beam_size)

    for prediction in predictions_list:
        results += decoder_func(prediction, labels=labels)
    return results


def __gather_transcripts(transcript_list: list, transcript_len_list: list, labels: list) -> list:
    results = []
    labels_map = dict([(i, labels[i]) for i in range(len(labels))])
    # iterate over workers
    for t, ln in zip(transcript_list, transcript_len_list):
        # iterate over batch
        t_lc = t.long().cpu()
        ln_lc = ln.long().cpu()
        for ind in range(t.shape[0]):
            tgt_len = ln_lc[ind].item()
            target = t_lc[ind][:tgt_len].numpy().tolist()
            reference = ''.join([labels_map[c] for c in target])
            results.append(reference)
    return results


def process_evaluation_batch(tensors: dict, global_vars: dict, labels: list):
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
            global_vars['predictions'] += __gather_predictions(v, labels=labels, decoder_type='ctc', beam_size=0)
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list, transcript_len_list, labels=labels)


def process_transducer_evaluation_batch(
    tensors: dict, global_vars: dict, labels: list, decoder_type: str = 'ctc', beam_size: int = 1
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
                v, labels=labels, decoder_type=decoder_type, beam_size=beam_size
            )
            print("Predictions : ", global_vars['predictions'][-1])
        elif kv.startswith('transcript_length'):
            transcript_len_list = v
        elif kv.startswith('transcript'):
            transcript_list = v
        elif kv.startswith('output'):
            global_vars['logits'] += v

    global_vars['transcripts'] += __gather_transcripts(transcript_list, transcript_len_list, labels=labels)
    print("Reference : ", global_vars['transcripts'][-1])


def process_evaluation_epoch(global_vars: dict, eval_metric='WER', tag=None):
    """
    Calculates the aggregated loss and WER across the entire evaluation dataset
    """
    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    hypotheses = global_vars['predictions']
    references = global_vars['transcripts']

    eval_metric = eval_metric.upper()
    if eval_metric not in {'WER', 'CER'}:
        raise ValueError('eval_metric must be \'WER\' or \'CER\'')
    use_cer = True if eval_metric == 'CER' else False

    wer = word_error_rate(hypotheses=hypotheses, references=references, use_cer=use_cer)

    if tag is None:
        logging.info(f"==========>>>>>>Evaluation Loss: {eloss}")
        logging.info(f"==========>>>>>>Evaluation {eval_metric}: " f"{wer * 100 : 5.2f}%")
        return {"Evaluation_Loss": eloss, f"Evaluation_{eval_metric}": wer}
    else:
        logging.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss}")
        logging.info(f"==========>>>>>>Evaluation {eval_metric} {tag}: " f"{wer * 100 : 5.2f}%")
        return {
            f"Evaluation_Loss_{tag}": eloss,
            f"Evaluation_{eval_metric}_{tag}": wer,
        }


def post_process_predictions(predictions, labels, decoder_type: str = 'ctc', beam_size: int = 1):
    if decoder_type not in ('ctc', *ALLOWED_RNNT_DECODING_SCHEMES):
        raise ValueError('`decoder_type` must be either `ctc` or {}'.format(str(ALLOWED_RNNT_DECODING_SCHEMES)))

    return __gather_predictions(predictions, labels=labels, decoder_type=decoder_type, beam_size=beam_size)


def post_process_transcripts(transcript_list, transcript_len_list, labels):
    return __gather_transcripts(transcript_list, transcript_len_list, labels=labels)


def process_classification_evaluation_batch(tensors: dict, global_vars: dict, top_k: list = 1):
    """
    Creates a dictionary holding the results from a batch of samples
    """
    if 'EvalLoss' not in global_vars.keys():
        global_vars['EvalLoss'] = []
    if 'batchsize' not in global_vars.keys():
        global_vars['batchsize'] = []

    if isinstance(top_k, int):
        top_k = [top_k]

    top_k = sorted(top_k)

    for k in top_k:
        if f'CorrectCount@{k}' not in global_vars.keys():
            global_vars[f'CorrectCount@{k}'] = []

    logits = None
    labels = None

    for kv, v in tensors.items():
        if kv.startswith('loss'):
            global_vars['EvalLoss'] += __gather_losses(v)
        elif kv.startswith('logits'):
            logits = torch.cat(v, 0)  # if len(v) > 1 else v
        elif kv.startswith('label'):
            labels = torch.cat(v, 0)  # if len(v) > 1 else v

    batch_size = labels.size(0)
    global_vars['batchsize'] += [batch_size]

    with torch.no_grad():
        topk_acc = classification_accuracy(logits, labels, top_k=top_k)

    for k, acc in zip(top_k, topk_acc):
        # Accuracy is provided as a percentage, we require the count of correct samples
        # Therefore multiply by batch size to get count of correctly predicted samples
        global_vars[f'CorrectCount@{k}'] += [acc * batch_size]


def process_classification_evaluation_epoch(global_vars: dict, eval_metric=None, tag=None):
    """
    Calculates the aggregated loss and WER across the entire evaluation dataset
    """
    if eval_metric is None:
        eval_metric = [1]

    if type(eval_metric) not in (list, tuple):
        eval_metric = [eval_metric]

    top_k = eval_metric

    eloss = torch.mean(torch.stack(global_vars['EvalLoss'])).item()
    batch_sizes = global_vars['batchsize']
    total_num_samples = torch.tensor(batch_sizes).sum().double()

    topk_accs = []
    for k in top_k:
        correct_counts = torch.tensor(global_vars[f'CorrectCount@{k}'])
        topk_acc = correct_counts.sum().double() / total_num_samples
        topk_accs.append(topk_acc)

    if tag is None:
        tag = ''

    logs = {f"Evaluation_Loss {tag}": eloss}

    logging.info(f"==========>>>>>>Evaluation Loss {tag}: {eloss:.3f}")
    for k, acc in zip(top_k, topk_accs):
        logging.info(f"==========>>>>>>Evaluation Accuracy Top@{k} {tag}: {acc * 100.:3.4f}")
        logs[f'Evaluation_Accuracy_Top@{k} {tag}'] = acc * 100.0

    return logs
