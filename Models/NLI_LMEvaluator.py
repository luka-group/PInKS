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
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

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

        # self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

        assert self.hparams['model_setup.tuned_model_path'] is not None
        assert 'roberta' in self.hparams["model_setup.model_name"]
        self.tuned_lm = ModifiedLMModule.load_from_checkpoint(self.hparams['model_setup.tuned_model_path'])

        # self._test_method()
        self.embedder.roberta = self.tuned_lm.model.roberta

    # def _test_method(self):
    #     loader = self.train_dataloader()
    #     batch = next(iter(loader))
    #     input_ids = batch["input_ids"]
    #     attention_mask = batch["attention_mask"]
    #     token_type_ids = None
    #     position_ids = None
    #     head_mask = None
    #     inputs_embeds = None
    #     labels = batch['labels']
    #     output_attentions = None
    #     output_hidden_states = None
    #     return_dict = None
    #     return_dict = return_dict if return_dict is not None else self.embedder.config.use_return_dict
    #
    #     IPython.embed()
    #     exit()
    #     orig_output = self.embedder.roberta(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #     modf_output = self.tuned_lm.model(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #         head_mask=head_mask,
    #         inputs_embeds=inputs_embeds,
    #         output_attentions=output_attentions,
    #         output_hidden_states=output_hidden_states,
    #         return_dict=return_dict,
    #     )
    #
    #
    #     exit()
    #


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

    # logger.info(f'Zero shot results')
    # _module.extra_tag = 'zero'
    # trainer.test(_module)

    logger.info('Tuning')
    _module.extra_tag = 'fit'
    if config['train_setup']['do_train'] is True:
        trainer.fit(_module)

    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    trainer.test(_module)


if __name__ == '__main__':
    main()
