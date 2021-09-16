import os

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Models.Modules.ModifiedLangModelingModule import ModifiedLMModule
from Models.Modules.BaseNLIModule import NLIModule
# from Models.Utils import flatten_config
from Models import Utils

import logging
logger = logging.getLogger(__name__)


class NLIModuleWithTunedLM(NLIModule):
    def __init__(self, config):
        # skip running parent's init function and just run the grandparent's
        super(NLIModuleWithTunedLM, self).__init__(config)

        # self.hparams = Utils.flatten_config(config)
        # if self.logger is not None:
        #     self.logger.log_hyperparams(self.hparams)
        # self.extra_tag = ''
        # self.save_hyperparameters()
        #
        # HOME_DIR = os.path.expanduser('~')
        # self.tokenizer = AutoTokenizer.from_pretrained(
        #     self.hparams["model_setup.model_name"],
        #     cache_dir=f"{HOME_DIR}/model_cache",
        #     use_fast=False
        # )
        #
        # self.embedder = AutoModelForSequenceClassification.from_pretrained(
        #     self.hparams["model_setup.model_name"],
        #     cache_dir="f{HOME_DIR}/model_cache"
        # )

        assert self.hparams['model_setup.tuned_model_path'] is not None
        assert 'roberta' in self.hparams["model_setup.model_name"]
        self.tuned_lm = ModifiedLMModule.load_from_checkpoint(self.hparams['model_setup.tuned_model_path'])
        logger.info(f'Replacing the weights')
        self.embedder.roberta.load_state_dict(self.tuned_lm.model.roberta.state_dict())
        logger.info(f'Deleting redundant model')
        del self.tuned_lm

