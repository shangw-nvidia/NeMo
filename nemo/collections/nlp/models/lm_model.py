# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nemo.collections.common.losses import AggregatorLoss, CrossEntropyLoss
from nemo.collections.common.tokenizers.bert_tokenizer import NemoBertTokenizer
from nemo.collections.nlp.data.lm_bert_dataset import BertPretrainingPreprocessedDataloader
from nemo.collections.nlp.losses import SmoothedCrossEntropyLoss
from nemo.collections.nlp.modules.common import SequenceClassifier, TokenClassifier
from nemo.collections.nlp.modules.common.huggingface.bert import BertEncoder
from nemo.core.classes import typecheck
from nemo.core.classes.modelPT import ModelPT
from nemo.core.neural_types import NeuralType
from nemo.utils.decorators import experimental

__all__ = ['BERTLMModel']


@experimental
class BERTLMModel(ModelPT):
    """
    BERT LM model pretraining.
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        return self.bert_model.input_types

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        return {
            'mlm_logits': self.mlm_classifier.output_types['logits'],
            'nsp_logits': self.nsp_classifier.output_types['logits'],
        }

    def __init__(self, num_classes: int, pretrained_model_name: Optional[str] = 'bert-base-uncased'):
        """
        Args:
            num_classes: output vocabulary size of language model
            pretrained_model_name: BERT model name 
        """
        super().__init__()
        self.bert_model = BertEncoder.from_pretrained(pretrained_model_name)
        self.hidden_size = self.bert_model.config.hidden_size
        self.tokenizer = NemoBertTokenizer(pretrained_model=pretrained_model_name)
        self.mlm_classifier = TokenClassifier(
            hidden_size=self.hidden_size,
            num_classes=num_classes,
            activation='gelu',
            log_softmax=True,
            use_transformer_init=True,
        )

        self.nsp_classifier = SequenceClassifier(
            hidden_size=self.hidden_size,
            num_classes=2,
            num_layers=2,
            log_softmax=False,
            activation='tanh',
            use_transformer_init=True,
        )

        self.mlm_loss = SmoothedCrossEntropyLoss()
        self.nsp_loss = CrossEntropyLoss()
        self.agg_loss = AggregatorLoss(num_inputs=2)

        # # tie weights of MLM softmax layer and embedding layer of the encoder
        if (
            self.mlm_classifier.mlp.last_linear_layer.weight.shape
            != self.bert_model.embeddings.word_embeddings.weight.shape
        ):
            raise ValueError("Final classification layer does not match embedding layer.")
        self.mlm_classifier.mlp.last_linear_layer.weight = self.bert_model.embeddings.word_embeddings.weight
        # create extra bias

        # This will be set by setup_training_datai
        self.__train_dl = None
        # This will be set by setup_validation_data
        self.__val_dl = None
        # This will be set by setup_test_data
        self.__test_dl = None
        # This will be set by setup_optimization
        self.__optimizer = None

    @typecheck()
    def forward(self, input_ids, token_type_ids, attention_mask):
        """
        No special modification required for Lightning, define it as you normally would
        in the `nn.Module` in vanilla PyTorch.
        """
        hidden_states = self.bert_model(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        mlm_logits = self.mlm_classifier(hidden_states=hidden_states)
        nsp_logits = self.nsp_classifier(hidden_states=hidden_states)
        return mlm_logits, nsp_logits

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        mlm_logits, nsp_logits = self.forward(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        mlm_loss = self.mlm_loss(logits=mlm_logits, labels=output_ids, output_mask=output_mask)
        nsp_loss = self.nsp_loss(logits=nsp_logits, labels=labels)

        loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        input_ids, input_type_ids, input_mask, output_ids, output_mask, labels = batch
        mlm_logits, nsp_logits = self.forward(
            input_ids=input_ids, token_type_ids=input_type_ids, attention_mask=input_mask
        )

        mlm_loss = self.mlm_loss(logits=mlm_logits, labels=output_ids, output_mask=output_mask)
        nsp_loss = self.nsp_loss(logits=nsp_logits, labels=labels)

        loss = self.agg_loss(loss_1=mlm_loss, loss_2=nsp_loss)

        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}

    def setup_training_data(self, train_data_layer_params: Optional[Dict]):
        if 'shuffle' not in train_data_layer_params:
            train_data_layer_params['shuffle'] = True
        self.__train_dl = self.__setup_dataloader(
            train_data_layer_params['train_data'],
            train_data_layer_params['max_pred_length'],
            train_data_layer_params['batch_size'],
        )

    def setup_validation_data(self, val_data_layer_params: Optional[Dict]):
        if 'shuffle' not in val_data_layer_params:
            val_data_layer_params['shuffle'] = False
        self.__val_dl = self.__setup_dataloader(
            val_data_layer_params['eval_data'],
            val_data_layer_params['max_pred_length'],
            val_data_layer_params['batch_size'],
        )

    def setup_test_data(self, test_data_layer_params: Optional[Dict]):
        pass

    def setup_optimization(self, optim_params: Optional[Dict], optimizer='adam'):
        if optimizer == 'adam':
            self.__optimizer = torch.optim.Adam(
                self.parameters(),
                lr=optim_params['lr'],
                weight_decay=optim_params.get('weight_decay', 0),
                betas=optim_params.get('betas', (0.9, 0.999)),
                eps=optim_params.get('eps', 1e-08),
            )
        elif optimizer == 'adam_w':
            self.__optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optim_params['lr'],
                weight_decay=optim_params.get('weight_decay', 0),
                betas=optim_params.get('betas', (0.9, 0.999)),
                eps=optim_params.get('eps', 1e-08),
            )
        else:
            raise NotImplementedError()

    def __setup_dataloader(
        self, dataset, max_pred_length, batch_size=64,
    ):
        if os.path.isdir(dataset):
            files = [os.path.join(dataset, f) for f in os.listdir(dataset) if os.path.isfile(os.path.join(dataset, f))]
        else:
            files = [dataset]
        files.sort()
        dl = BertPretrainingPreprocessedDataloader(
            data_files=files, max_pred_length=max_pred_length, batch_size=batch_size
        )
        # return torch.utils.data.DataLoader(
        #     dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        # )
        return dl

    def configure_optimizers(self):
        return self.__optimizer

    def train_dataloader(self):
        return self.__train_dl

    def val_dataloader(self):
        return self.__val_dl

    @classmethod
    def from_pretrained(cls, name: str):
        pass

    def export(self, **kwargs):
        pass

    def save_to(self, save_path: str):
        pass

    @classmethod
    def restore_from(cls, restore_path: str):
        pass

    @classmethod
    def list_available_models(cls) -> Optional[Dict[str, str]]:
        pass