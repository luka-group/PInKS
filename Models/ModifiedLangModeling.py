import itertools
import pathlib
from functools import partial

import IPython
import hydra
from typing import *
from itertools import cycle
import os

import omegaconf
import torch
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
# from loguru import logger
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification, AutoModelForMaskedLM
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score


# from LMBenchmarkEvaluator import BaseEvaluationModule
from ModifiedLMData import LMDataModule
from Utils import ClassificationDataset

import logging
logger = logging.getLogger(__name__)


class ModifiedLMModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self._setup_hparams(config)

        self.model = AutoModelForMaskedLM.from_pretrained(
            config['lm_module']["model_name_or_path"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

    def _setup_hparams(self, config, prefix: str = ''):
        key = lambda k: (f'{prefix}.' if prefix != '' else '') + f'{k}'
        for k, v in dict(config).items():
            if any([isinstance(v, t) for t in [int, float, str, bool, torch.Tensor]]):
                self.hparams[key(k)] = v
            elif v is None:
                self.hparams[key(k)] = ''
            elif isinstance(v, omegaconf.dictconfig.DictConfig):
                # [self.hparams.__setitem__(f'{k}.{kk}', vv) for kk, vv in v.items()]
                self._setup_hparams(v, prefix=k)
            else:
                raise ValueError(f'invalid config type: [{k}]={v} with type {type(v)}')

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True)

    def configure_optimizers(self):
        logger.info(f'KEYS: {self.hparams.keys()}')
        optimizer = AdamW(self.parameters(),
                          self.hparams['lm_module.learning_rate'],
                          betas=(self.hparams['lm_module.adam_beta1'],
                                 self.hparams['lm_module.adam_beta2']),
                          eps=self.hparams['lm_module.adam_epsilon'], )
        return optimizer


@hydra.main(config_path='../Configs/modified_lm_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):

    # ------------
    # data
    # ------------
    data_module = LMDataModule(config)

    # ------------
    # model
    # ------------
    lmmodel = ModifiedLMModule(config)

    # trainer = pl.Trainer(fast_dev_run=1)
    # data_module.setup('train')
    # loader = data_module.train_dataloader()
    # batch = next(iter(loader))
    # out = lmmodel.training_step(batch, 0)
    # IPython.embed()
    # exit()


    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gradient_clip_val=0,
        # gpus=config['hardware']['gpus'],
        # gpus='',
        max_epochs=1,
        min_epochs=1,
        resume_from_checkpoint=None,
        distributed_backend=None,
    )

    trainer.fit(lmmodel, datamodule=data_module)
    # trainer.test(lmmodel, datamodule=data_module)


if __name__ == '__main__':
    main()
