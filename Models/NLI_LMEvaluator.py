import functools
import logging
from typing import Optional

import IPython
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ModifiedLangModeling import ModifiedLMModule
from NLIEvaluator import NLIModule
from Utils import config_to_hparams

logger = logging.getLogger(__name__)


class NLILMModule(NLIModule):
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

        # self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

        assert self.hparams['model_setup.tuned_model_path'] is not None
        assert 'roberta' in self.hparams["model_setup.model_name"]
        tuned_lm = ModifiedLMModule.load_from_checkpoint(self.hparams['model_setup.tuned_model_path'])
        self.embedder.roberta = tuned_lm.model


class Weak2CqNliData(pl.LightningDataModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.hparams = config_to_hparams(config)
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.data_collator = None

    def setup(self, stage: Optional[str] = None):
        tokenizer = functools.partial(
            AutoTokenizer.from_pretrained(
                self.hparams['model_setup.model_name'],
                cache_dir="/nas/home/qasemi/model_cache",
            ),
            add_special_tokens=True,
            padding='max_length',
            max_length=self.hparams["model_setup.max_length"],
            return_tensors='np',
            return_token_type_ids=True,
            truncation=True,
        )

        # data_files =
        #     'test': self.hparams[]
        # }
        # data_files["train"] = self.config.data_module.train_file
        # data_files["validation"] = self.config.data_module.validation_file
        train_dataset = load_dataset('csv', data_files={'train': self.hparams['weak_cq_path']},)
        column_names = train_dataset["train"].column_names
        train_dataset.map(
            function=self._tokenize_weak_cq,
            remove_columns=column_names,
            batched=True,
            num_proc=self.hparams['data_module.preprocessing_num_workers'],
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

    @staticmethod
    def _tokenize_weak_cq(_examples, _tokenizer):
        _examples["text"] = [f"{act} </s></s> {pred}"
                             for act, pred in zip(_examples["action"], _examples['precondition'])]
        return {
            **_tokenizer(_examples["text"]),
            "labels": torch.LongTensor(_examples["label"]),
            "text": _examples['text'],
        }


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLILMModule(config)

    loader = _module.train_dataloader()
    batch = next(iter(loader))
    IPython.embed()
    exit()
    # IPython.embed()
    # exit()
    output = _module.forward(batch)
    output = _module.training_step(batch, 0)

    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['train_setup']['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config['train_setup']['warmup_steps'],
        resume_from_checkpoint=None,
        distributed_backend=None,
    )

    if config['train_setup']['do_train'] is True:
        trainer.fit(_module)

    trainer.test(_module)


if __name__ == '__main__':
    main()
