"""
Author: Awni Hannun
This is an example CTC decoder written in Python. The code is
intended to be a simple example and is not designed to be
especially efficient.
The algorithm is a prefix beam search for a model trained
with the CTC loss function.
For more details checkout either of these references:
  https://distill.pub/2017/ctc/#inference
  https://arxiv.org/abs/1408.2873
"""

import math
import os
from typing import Dict, Tuple

import numpy as np

try:
    from nemo.collections.asr.parts import numba_utils

    HAVE_NUMBA = True
except (ImportError, ModuleNotFoundError):
    HAVE_NUMBA = False

try:
    import joblib

    HAVE_JOBLIB = True
except (ImportError, ModuleNotFoundError):
    HAVE_JOBLIB = False


def log_sum_exp(a, b):
    """
    Stable log sum exp.
    """
    if HAVE_NUMBA:
        return numba_utils.log_sum_exp(a, b)
    else:
        return max(a, b) + math.log1p(math.exp(-abs(a - b)))


def remove_padded_timesteps(mask_cumsum):
    if HAVE_NUMBA:
        return numba_utils.remove_padded_timesteps(mask_cumsum)
    else:
        T, U, V = mask_cumsum.shape

        for t in range(T - 1, 0, -1):
            if mask_cumsum[t, 0, -1] == 0.0:
                T = T - 1
            else:
                break

        return T


def rnnt_beam_decode_static(x, blank_idx, beam_size, parallel=True):
    if HAVE_JOBLIB and parallel:
        return _rnnt_beam_decode_static_joblib(x, blank_idx, beam_size)
    else:
        return _rnnt_beam_decode_static_sequential(x, blank_idx, beam_size)


def _rnnt_beam_decode_static_joblib(x, blank_idx, beam_size):
    n_jobs = min(os.cpu_count(), x.shape[0])

    with joblib.Parallel(n_jobs=n_jobs, verbose=0) as parallel:
        results = parallel(
            joblib.delayed(decode_static)(x[batch_idx], blank_idx, beam_size) for batch_idx in range(x.shape[0])
        )
        results = [res[0] for res in results]

    return results


def _rnnt_beam_decode_static_sequential(x, blank_idx, beam_size):
    results = [
        decode_static(x[batch_idx], blank_idx, beam_size)[0]  # require only sequence, not log prob
        for batch_idx in range(x.shape[0])
    ]
    return results


def decode_static(log_probs, blank, beam_size=1):
    """
    Decode best prefix in the RNN Transducer. This decoder is static, it does
    not update the next step distribution based on the previous prediction. As
    such it looks for hypotheses which are length U.
    """
    # [T, U, V]
    T, U, V = log_probs.shape

    # removed padded timesteps
    mask_cumsum = log_probs.cumsum(axis=-1)
    T = remove_padded_timesteps(mask_cumsum)

    beam = [((), 0.0)]

    for i in range(T + U - 2):
        new_beam = {}
        for hyp, score in beam:
            u = len(hyp)
            t = i - u
            for v in range(V):
                if v == blank:
                    if t < T - 1:
                        new_hyp = hyp
                        new_score = score + log_probs[t, u, v]
                    else:
                        new_hyp = hyp
                        new_score = score + log_probs[-1, u, v]

                elif t < T - 1 and u < U - 1:
                    new_hyp = hyp + (v,)
                    new_score = score + log_probs[t, u, v]
                else:
                    continue

                old_score = new_beam.get(new_hyp, None)
                if old_score is not None:
                    new_beam[new_hyp] = log_sum_exp(old_score, new_score)
                else:
                    new_beam[new_hyp] = new_score

        new_beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    hyp, score = beam[0]
    return hyp, score + log_probs[-1, -1, blank]


def rnnt_beam_decode_dynamic(packed_results, blank_idx, parallel=True):
    if HAVE_JOBLIB and parallel:
        return _rnnt_beam_decode_dynamic_joblib(packed_results, blank_idx)
    else:
        return _rnnt_beam_decode_dynamic_sequential(packed_results, blank_idx)


def _rnnt_beam_decode_dynamic_joblib(packed_results, blank_idx):
    n_jobs = min(os.cpu_count(), packed_results.shape[0])

    with joblib.Parallel(n_jobs=n_jobs, verbose=0) as parallel:
        results = parallel(
            joblib.delayed(decode_dynamic)(packed_results[batch_idx], blank_idx)
            for batch_idx in range(packed_results.shape[0])
        )

    return results


def _rnnt_beam_decode_dynamic_sequential(packed_results, blank_idx):
    results = [
        decode_dynamic(packed_results[batch_idx], blank_idx)  # require only sequence, not log prob
        for batch_idx in range(packed_results.shape[0])
    ]
    return results


def decode_dynamic(packed_result, blank):
    result = [char for char in packed_result
              if char != blank]
    return result
