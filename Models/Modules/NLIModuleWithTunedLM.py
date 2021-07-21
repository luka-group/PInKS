from transformers import AutoTokenizer, AutoModelForSequenceClassification

from Models.Modules.ModifiedLangModelingModule import ModifiedLMModule
from Models.Modules.BaseNLIModule import NLIModule
from Models.Utils import config_to_hparams

import logging
logger = logging.getLogger(__name__)


class NLIModuleWithTunedLM(NLIModule):
    def __init__(self, config):
        # skip running parent's init function and just run the grandparent's
        super(NLIModule, self).__init__()

        self.hparams = config_to_hparams(config)
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache",
            use_fast=False
        )

        # self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

        assert self.hparams['model_setup.tuned_model_path'] is not None
        assert 'roberta' in self.hparams["model_setup.model_name"]
        self.tuned_lm = ModifiedLMModule.load_from_checkpoint(self.hparams['model_setup.tuned_model_path'])

        # self._test_method()
        # self.embedder.roberta = self.tuned_lm.model.roberta
        logger.info(f'Replacing the weights')
        self.embedder.roberta.load_state_dict(self.tuned_lm.model.roberta.state_dict())
        logger.info(f'Deleting redundant model')
        del self.tuned_lm

