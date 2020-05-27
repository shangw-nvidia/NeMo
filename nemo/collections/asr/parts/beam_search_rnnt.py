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

import collections
import math
from typing import Dict, Tuple

import numba
import numpy as np

NEG_INF = -float("inf")


def make_new_beam():
    def fn():
        return (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def decode(probs, blank, beam_size=4):
    """
    Performs inference for the given output probabilities.
    Arguments:
      probs: The output probabilities (e.g. post-softmax) for each
        time step. Should be an array of shape (time x output dim).
      beam_size (int): Size of the beam to use during inference.
      blank (int): Index of the CTC blank label.
    Returns the output label sequence and the corresponding negative
    log-likelihood estimated by the decoder.
    """
    T, S = probs.shape
    # probs = np.log(probs)

    # Elements in the beam are (prefix, (p_blank, p_no_blank))
    # Initialize the beam with the empty sequence, a probability of
    # 1 for ending in blank and zero for ending in non-blank
    # (in log space).
    beam = [(tuple(), (0.0, NEG_INF))]

    for t in range(T):  # Loop over time

        # A default dictionary to store the next step candidates.
        next_beam = make_new_beam()

        for s in range(S):  # Loop over vocab
            p = probs[t, s]

            # The variables p_b and p_nb are respectively the
            # probabilities for the prefix given that it ends in a
            # blank and does not end in a blank at this time step.
            for prefix, (p_b, p_nb) in beam:  # Loop over beam

                # If we propose a blank the prefix doesn't change.
                # Only the probability of ending in blank gets updated.
                if s == blank:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_b = logsumexp(n_p_b, p_b + p, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)
                    continue

                # Extend the prefix by the new character s and add it to
                # the beam. Only the probability of not ending in blank
                # gets updated.
                end_t = prefix[-1] if prefix else None
                n_prefix = prefix + (s,)
                n_p_b, n_p_nb = next_beam[n_prefix]
                if s != end_t:
                    n_p_nb = logsumexp(n_p_nb, p_b + p, p_nb + p)
                else:
                    # We don't include the previous probability of not ending
                    # in blank (p_nb) if s is repeated at the end. The CTC
                    # algorithm merges characters not separated by a blank.
                    n_p_nb = logsumexp(n_p_nb, p_b + p)

                # *NB* this would be a good place to include an LM score.
                next_beam[n_prefix] = (n_p_b, n_p_nb)

                # If s is repeated at the end we also update the unchanged
                # prefix. This is the merging case.
                if s == end_t:
                    n_p_b, n_p_nb = next_beam[prefix]
                    n_p_nb = logsumexp(n_p_nb, p_nb + p)
                    next_beam[prefix] = (n_p_b, n_p_nb)

        # Sort and trim the beam before moving on to the
        # next time-step.
        beam = sorted(next_beam.items(), key=lambda x: logsumexp(*x[1]), reverse=True)
        beam = beam[:beam_size]

    best = beam[0]
    return best[0], -logsumexp(*best[1])


@numba.jit(nopython=True)
def log_sum_exp(a, b):
    """
    Stable log sum exp.
    """
    return max(a, b) + math.log1p(math.exp(-abs(a-b)))


@numba.jit(nopython=True)
def _remove_padded_timesteps(mask_cumsum):
    T, U, V = mask_cumsum.shape

    for t in range(T - 1, 0, -1):
        if mask_cumsum[t, 0, -1] == 0.0:
            T -= 1
        else:
            break

    return T


# @numba.njit()
# def decode_static(log_probs, blank, beam_size=1):
#     """
#     Decode best prefix in the RNN Transducer. This decoder is static, it does
#     not update the next step distribution based on the previous prediction. As
#     such it looks for hypotheses which are length U.
#     """
#     # [T, U, V]
#     T, U, V = log_probs.shape
#
#     # removed padded timesteps
#     mask_cumsum = log_probs.cumsum(axis=-1)
#     T = _remove_padded_timesteps(mask_cumsum)
#
#     beam = [((), 0.0)]
#
#     for i in range(T + U - 2):
#         new_beam = {}
#         for hyp, score in beam:
#             u = len(hyp)
#             t = i - u
#             for v in range(V):
#                 if v == blank:
#                     if t < T - 1:
#                         new_hyp = hyp
#                         new_score = score + log_probs[t, u, v]
#                 elif u < U - 1:
#                     new_hyp = hyp + (v,)
#                     new_score = score + log_probs[t, u, v]
#                 else:
#                     continue
#
#                 old_score = new_beam.get(new_hyp, None)
#                 if old_score is not None:
#                     new_beam[new_hyp] = log_sum_exp(old_score, new_score)
#                 else:
#                     new_beam[new_hyp] = new_score
#
#         new_beam = sorted(new_beam.items(), key=lambda x: x[1], reverse=True)
#         beam = new_beam[:beam_size]
#
#     hyp, score = beam[0]
#     return hyp, score + log_probs[-1, -1, blank]


# @numba.njit()
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
    T = _remove_padded_timesteps(mask_cumsum)

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
