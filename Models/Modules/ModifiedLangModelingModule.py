import os

import pytorch_lightning as pl
from transformers import AutoModelForMaskedLM, AdamW

from Models.Utils import flatten_config

import logging
logger = logging.getLogger(__name__)


class ModifiedLMModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = flatten_config(config)
        logger.info(f'hparams: {self.hparams}')
        self.save_hyperparameters()

        HOME_DIR = os.path.expanduser('~')
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.hparams['lm_module.model_name_or_path'],
            cache_dir=f"{HOME_DIR}/model_cache"
        )

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