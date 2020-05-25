# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import nemo
from .parts.numba_utils import rnnt_beam_decode
from .parts.rnn import label_collate
from .rnnt import RNNTDecoder, RNNTJoint
from nemo.backends.pytorch.nm import NonTrainableNM, TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

logging = nemo.logging


__all__ = ['GreedyRNNTDecoderInfer', 'GreedyRNNTDecoder']


@torch.jit.script
def _greedy_decode(
    x: torch.Tensor, out_len: int, results: torch.Tensor, batch_idx: int, max_symbols: int, blank_index: int
):
    x = x.float()

    for time_idx in range(out_len):
        not_blank = True
        symbols_added = 0

        while not_blank and (max_symbols < 0 or symbols_added < max_symbols):
            logp = x[time_idx, symbols_added, :]  # [K]

            # get index k, of max prob
            v, k = logp.max(0)
            k = k.item()

            if k == blank_index:
                not_blank = False
            else:
                # chars.append(k)
                results[batch_idx, time_idx, symbols_added] = k
            symbols_added += 1

    return results


# @torch.jit.script
# def _greedy_decode_v2(x: torch.Tensor, out_seq: torch.Tensor, max_symbols: int, blank_index: int):
#     # x = [B, T, U, K] -> [B, T, max_symbols, K]
#     x = x[:, :, :max_symbols, :]
#
#     if x.dtype != torch.float32:
#         x = x.float()
#
#     # symbols : [B, T, max_symbols]
#     k, symbols = x.max(-1)
#
#     # At each timestep, if blank occurs, remaining of all symbols should also be blank for that timestep
#     # To broadcast this, we first take create a bool mask for all locations of blanks
#     # Then we take cumulative sum to obtain a 1 or more at the first or later instances of blanks
#     # Then we cast it to binary with a > 0 check. This avoids filling values before the first blank
#     # with blank tokens.
#     blank_mask = (symbols == blank_index).cumsum(-1)
#     blank_mask = (blank_mask > 0)
#
#     # Mask out entries after out_seq timesteps
#     # This ensures that for each sample in batch, we only predict
#     # tokens as long as the length of the original sequence without padding
#     time_mask = torch.full([x.size(0), x.size(1), 1], fill_value=0, dtype=torch.bool, device=x.device)
#
#     for seq_id in range(out_seq.size(0)):
#         seq_len = out_seq[seq_id]
#         time_mask[seq_id, seq_len:, :] = 1
#
#     # bitwise or the masks to combine them
#     blank_mask = blank_mask.bitwise_or(time_mask)
#
#     symbols.masked_fill_(blank_mask, blank_index)
#     return symbols


def _greedy_decode_v2(x: torch.Tensor, out_seq: torch.Tensor, max_symbols: int, blank_index: int):
    # x = [B, T, U, K] -> [B, T, max_symbols, K]
    x = x[:, :, :max_symbols, :]

    if x.dtype != torch.float32:
        x = x.float()

    # symbols : [B, T, max_symbols, vocab + 1]
    x = x.log_softmax(dim=-1)

    # Mask out entries after out_seq timesteps
    # This ensures that for each sample in batch, we only predict
    # tokens as long as the length of the original sequence without padding
    time_mask = torch.full([x.size(0), x.size(1), 1, 1], fill_value=0, dtype=torch.bool, device=x.device)

    for seq_id in range(out_seq.size(0)):
        seq_len = out_seq[seq_id]
        time_mask[seq_id, seq_len:, :, :] = 1

    x.masked_fill_(time_mask, 0.0)

    return x


class GreedyRNNTDecoder(NonTrainableNM):
    """A greedy transducer decoder.
    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"log_probs": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),})}
        return {
            "joint_output": NeuralType(('B', 'T', 'D', 'D'), LogitsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(self, blank_index: int, max_symbols_per_step: int = 30, log_normalize: bool = False):
        super().__init__()

        self._blank_index = blank_index
        self._SOS = -1  # Start of single index
        self.log_normalize = log_normalize

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError('`max_symbols_per_step` must be None or positive integer')

        if max_symbols_per_step is None:
            max_symbols_per_step = -1

        self.max_symbols = max_symbols_per_step

    @torch.no_grad()
    def forward(self, joint_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of sentences given an input batch.
        Args:
            x: A tensor of size (batch, features, timesteps).
            out_lens: list of int representing the length of each sequence
                output sequence.
        Returns:
            list containing batch number of sentences (strings).
        """
        log_probs = self._joint_step(joint_output, log_normalize=self.log_normalize)  # [B, T, U + 1, K + 1]
        # log_probs = log_probs.to('cpu')
        logitlen = encoded_lengths.to('cpu')  # .numpy()

        if self.max_symbols < 0:
            max_symbols = int(log_probs.size(2))
        else:
            max_symbols = self.max_symbols

        # create a result buffer of shape [B, T, max_symbols]
        # results = torch.full(
        #     log_probs.shape[:-1], fill_value=self._blank_index, dtype=torch.int32, device=log_probs.device
        # )

        # for batch_idx in range(log_probs.size(0)):
        #     inseq = log_probs[batch_idx, :, :, :]  # [T, U + 1, K + 1]
        #     out_len = logitlen[batch_idx]
        #     results = _greedy_decode(inseq, out_len, results, batch_idx, self.max_symbols, self._blank_index)

        results = _greedy_decode_v2(log_probs, logitlen, max_symbols, self._blank_index)

        # results = results.to(self._device)
        return results

    @torch.no_grad()
    def _joint_step(self, joint_output, log_normalize=False):
        """

        Args:
            joint_output: Input tensor [B, T, U + 1, K]
            log_normalize:

        Returns:
             logits of shape (B, T, K + 1)
        """
        logits = joint_output[:, :, : self.max_symbols, :]

        if not log_normalize:
            return logits

        logits = logits.float()
        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
        return probs


class GreedyRNNTDecoderInfer(NonTrainableNM):
    """A greedy transducer decoder.
    Args:
        blank_symbol: See `Decoder`.
        model: Model to use for prediction.
        max_symbols_per_step: The maximum number of symbols that can be added
            to a sequence in a single time step; if set to None then there is
            no limit.
        cutoff_prob: Skip to next step in search if current highest character
            probability is less than this.
    """

    @property
    @add_port_docs()
    def input_ports(self):
        """Returns definitions of module input ports.
        """
        # return {"log_probs": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag), 2: AxisType(ChannelTag),})}
        return {
            "encoder_output": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    @add_port_docs()
    def output_ports(self):
        """Returns definitions of module output ports.
        """
        # return {"predictions": NeuralType({0: AxisType(BatchTag), 1: AxisType(TimeTag)})}
        return {"predictions": NeuralType(('B', 'T'), PredictionsType())}

    def __init__(
        self, decoder_model: RNNTDecoder, joint_model: RNNTJoint, blank_index: int, max_symbols_per_step: int = 30,
    ):
        super().__init__()
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = -1  # Start of single index

        if max_symbols_per_step is None:
            max_symbols_per_step = -1

        self.max_symbols = max_symbols_per_step

    @torch.no_grad()
    def forward(self, encoder_output: torch.Tensor, encoded_lengths: torch.Tensor):
        """Returns a list of sentences given an input batch.
        Args:
            x: A tensor of size (batch, features, timesteps).
            out_lens: list of int representing the length of each sequence
                output sequence.
        Returns:
            list containing batch number of sentences (strings).
        """
        # Apply optional preprocessing
        encoder_output = encoder_output.transpose(1, 2)  # (B, T, D)
        logitlen = encoded_lengths.to('cpu').numpy()

        # create a result buffer of shape [B, T, max_symbols]
        results = torch.full(
            (*encoder_output.shape[:-1], self.max_symbols),
            fill_value=self._blank_index,
            dtype=torch.int32,
            device=self._device,
        )

        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        self.decoder.eval()
        self.joint.eval()

        # for batch_idx in range(encoder_output.size(0)):
        #     inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
        #     logitlen = encoded_lengths[batch_idx]
        #     results = self._greedy_decode(inseq, logitlen, results, batch_idx)

        inseq = encoder_output  # [batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
        logitlen = encoded_lengths  # [batch_idx]
        results = self._greedy_decode(inseq, logitlen, results, 0)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        results = results.to(self._device)
        return results

    # def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor, results: torch.Tensor, batch_idx: torch.Tensor):
    #     hidden = None
    #     label = []
    #     for time_idx in range(out_len):
    #         f = x[time_idx, :, :].unsqueeze(0)  # [1, 1, D]
    #
    #         not_blank = True
    #         symbols_added = 0
    #
    #         while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
    #             last_label = self._SOS if label == [] else label[-1]
    #             g, hidden_prime = self._pred_step(last_label, hidden)
    #             # print("g", g.shape)
    #             logp = self._joint_step(f, g, log_normalize=False)[0, :]
    #
    #             logp = logp.to('cpu').float()
    #
    #             # get index k, of max prob
    #             v, k = logp.max(0)
    #             k = k.item()
    #
    #             if k == self._blank_index:
    #                 not_blank = False
    #             else:
    #                 results[batch_idx, time_idx, symbols_added] = k
    #                 hidden = hidden_prime
    #
    #             symbols_added += 1
    #     return results

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor, results: torch.Tensor, batch_idx: int):
        # x : [B, T, D]
        # out_len : [B]
        hidden = None

        batch_size = x.size(0)
        max_out_len = out_len.max()

        # construct masks
        # mask out entries after out_seq timesteps
        time_mask = torch.full([x.size(0), x.size(1), 1], fill_value=0, dtype=torch.bool, device=x.device)

        for seq_id in range(out_len.size(0)):
            seq_len = out_len[seq_id]
            time_mask[seq_id, seq_len:, :] = 1

        blank_mask = torch.full([x.size(0)], fill_value=0, dtype=torch.bool, device=x.device)

        for time_idx in range(max_out_len):
            label = []

            f = x[:, time_idx, :].unsqueeze(1)  # [B, 1, D]

            # reset blank mask
            blank_mask.fill_(0)
            not_blank = True
            symbols_added = 0

            while not_blank and (self.max_symbols < 0 or symbols_added < self.max_symbols):
                last_label = self._SOS if len(label) == 0 else label
                g, hidden_prime = self._pred_step(last_label, hidden, batch_size=batch_size)
                # print("g", g.shape)

                # logp : [B, 1, K] -> [B, K]
                logp = self._joint_step(f, g, log_normalize=False)[:, :]
                logp = logp.float()

                # get index k, of max prob
                # k = [B]
                v, k = logp.max(-1)

                # k_is_blank : [B] (bool)
                k_is_blank = (k == self._blank_index)

                # update mask
                blank_mask = blank_mask.bitwise_or(k_is_blank)

                if blank_mask.all():
                    not_blank = False
                else:
                    # If the sample has now or previously predicted blank for this timestep,
                    # forcibly predict blank for remainder of timesteps too.
                    k.masked_fill_(blank_mask, self._blank_index)

                    # Update label for next step
                    label = k.unsqueeze(-1).long()  # [B, 1]

                    results[:, time_idx, symbols_added] = k
                    hidden = hidden_prime

                symbols_added += 1

        # Time mask the output so that extraneous timesteps are filled with blanks
        results.masked_fill_(time_mask, self._blank_index)

        return results

    def _pred_step(
        self,
        label: Union[torch.Tensor, int],
        hidden: Optional[torch.Tensor],
        batch_size: Optional[int] = None,
    ) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            label (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden (Optional torch.Tensor): RNN State vector
            batch_size (Optional torch.Tensor): Batch size of output

        Returns:
            g: (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """
        if isinstance(label, torch.Tensor):
            # label: [batch, 1]
            if label.dtype != torch.long:
                label = label.long()

        else:
            if label == self._SOS:
                return self.decoder.predict(None, hidden, add_sos=False, batch_size=batch_size)

            if label > self._blank_index:
                label -= 1

            label = label_collate([[label]]).to(self._device)

        # output: [B, 1, K]
        return self.decoder.predict(label, hidden, add_sos=False, batch_size=batch_size)

    def _joint_step(self, enc, pred, log_normalize=False):
        """

        Args:
            enc:
            pred:
            log_normalize:

        Returns:
             logits of shape (B, T, U, K + 1)
        """
        logits = self.joint.joint(enc, pred)[:, 0, 0, :]

        if not log_normalize:
            return logits

        probs = F.log_softmax(logits, dim=len(logits.shape) - 1)
        return probs
