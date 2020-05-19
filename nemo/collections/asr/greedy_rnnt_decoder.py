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
from .parts.rnn import label_collate
from .rnnt import RNNTDecoder, RNNTJoint
from nemo.backends.pytorch.nm import NonTrainableNM, TrainableNM
from nemo.core.neural_types import *
from nemo.utils.decorators import add_port_docs

logging = nemo.logging


__all__ = [
    'RNNTGreedyDecoder'
]


class RNNTGreedyDecoder(NonTrainableNM):
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
            "encoder_output": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation()),
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
        self,
        encoder_model: TrainableNM,
        decoder_model: RNNTDecoder,
        joint_model: RNNTJoint,
        blank_index: int,
        max_symbols_per_step: int = 30,
    ):
        super().__init__()
        self.encoder = encoder_model
        self.decoder = decoder_model
        self.joint = joint_model

        self._blank_index = blank_index
        self._SOS = -1  # Start of single index

        if max_symbols_per_step is not None and max_symbols_per_step <= 0:
            raise ValueError('`max_symbols_per_step` must be None or positive integer')

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

        # Preserve decoder and joint training state
        decoder_training_state = self.decoder.training
        joint_training_state = self.joint.training

        self.decoder.eval()
        self.joint.eval()

        output = []
        for batch_idx in range(encoder_output.size(0)):
            inseq = encoder_output[batch_idx, :, :].unsqueeze(1)  # [T, 1, D]
            logitlen = encoded_lengths[batch_idx]
            sentence = self._greedy_decode(inseq, logitlen)
            output.append(sentence)

        self.decoder.train(decoder_training_state)
        self.joint.train(joint_training_state)

        output = torch.tensor(output, device=self._device)

        return output

    def _greedy_decode(self, x: torch.Tensor, out_len: torch.Tensor):
        hidden = None
        label = []
        for time_idx in range(out_len):
            f = x[time_idx, :, :].unsqueeze(0)

            not_blank = True
            symbols_added = 0

            while not_blank and (self.max_symbols is None or symbols_added < self.max_symbols):
                last_label = self._SOS if label == [] else label[-1]
                g, hidden_prime = self._pred_step(last_label, hidden)
                logp = self._joint_step(f, g, log_normalize=False)[0, :]

                # get index k, of max prob
                v, k = logp.max(0)
                k = k.to('cpu').numpy()

                if k == self._blank_index:
                    not_blank = False
                else:
                    label.append(k)
                    hidden = hidden_prime

                symbols_added += 1

        return label

    def _pred_step(self, label: Union[torch.Tensor, int], hidden: Optional[torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            label (int/torch.Tensor): Label or "Start-of-Signal" token.
            hidden (Optional torch.Tensor): RNN State vector

        Returns:
            g: (B, U + 1, H)
            hid: (h, c) where h is the final sequence hidden state and c is
                the final cell state:
                    h (tensor), shape (L, B, H)
                    c (tensor), shape (L, B, H)
        """

        if label == self._SOS:
            return self.decoder.predict(None, hidden, add_sos=False)

        if label > self._blank_index:
            label -= 1

        label = label_collate([[label]]).to(self._device)
        return self.decoder.predict(label, hidden, add_sos=False)

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
