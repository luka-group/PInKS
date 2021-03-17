import pathlib

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
    AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score

import logging

# from LMBenchmarkEvaluator import BaseEvaluationModule
from Utils import ClassificationDataset

logger = logging.getLogger(__name__)


class NLIModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__(config)
        super().__init__()
        self.hparams = dict(config)
        self.root_path = pathlib.Path(__file__).parent.absolute()

        self.tokenizer = AutoTokenizer.from_pretrained(config["model"], cache_dir="model_cache", use_fast=False)

        self.loss = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean")

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            config["model"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

    def forward(self, batch):
        assert len(batch["input_ids"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["attention_mask"].shape) == 2, "LM only take two-dimensional input"
        assert len(batch["token_type_ids"].shape) == 2, "LM only take two-dimensional input"

        batch["token_type_ids"] = None if "roberta" in self.hparams["model"] else batch["token_type_ids"]

        results = self.embedder(input_ids=batch["input_ids"],
                               attention_mask=batch["attention_mask"],
                               token_type_ids=batch["token_type_ids"])

        logits, *_ = results
        # my_logits = torch.stack((logits[:, 0], logits[:, 1] + logits[:, 2]), dim=1)
        my_logits = logits
        return my_logits

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        loss_dict = {}
        if 'labels' in batch:
            loss_dict = self.validation_step(batch, batch_idx)
        return {
            k.replace('val_', 'test_'): v for k, v in loss_dict.items()
        }

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        self.logger.experiment.add_scalar('train_loss', loss)
        return {
            "loss": loss,
            "text": batch['text'],
        }

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch)
        loss = self.loss(logits, batch["labels"])
        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            'val_loss': loss,
            "val_batch_logits": logits,
            "val_batch_labels": batch["labels"],
            "text": batch['text'],
        }

    def validation_end(self, outputs):
        mytag = 'val'
        return self._collect_evaluation_results(outputs, mytag)

    def test_end(self, outputs):
        mytag = 'test'
        return self._collect_evaluation_results(outputs, mytag)

    def _collect_evaluation_results(self, outputs, mytag):
        _loss_mean = torch.stack([o[f'{mytag}_loss'] for o in outputs]).mean()
        _logits = torch.cat([o[f"{mytag}_batch_logits"] for o in outputs])
        _labels = torch.cat([o[f"{mytag}_batch_labels"] for o in outputs])
        val_acc = torch.sum(_labels == torch.argmax(_logits, dim=1)) / (_labels.shape[0] * 1.0)

        # f1_score = self._compute_f1_score(_labels, _logits)

        logger.info(f'{mytag}_acc={val_acc}, {mytag}_loss={_loss_mean}')

        self.logger.experiment.add_scalar(f'{mytag}_loss', _loss_mean)
        self.logger.experiment.add_scalar(f'{mytag}_acc', val_acc)

        # all_text =
        df = pd.DataFrame.from_dict({
            'predicted_label': torch.argmax(_logits, dim=1).detach().cpu().numpy(),
            'true_label': _labels.detach().cpu().numpy(),
        }, orient='columns')
        df['text'] = [s for o in outputs for s in o['text']]

        _f1_score = f1_score(y_true=df['true_label'], y_pred=df['predicted_label'], average='micro')
        self.logger.experiment.add_scalar(f'{mytag}_f1_macro', _f1_score)

        _conf_matrix = pd.DataFrame(
            confusion_matrix(y_true=df['true_label'], y_pred=df['predicted_label']),
            columns=[0, 1, 2],
            index=[0, 1, 2],
        )
        logger.info(f'{mytag}_confusion: \n{_conf_matrix}')
        # self.logger.experiment.add_(f'{mytag}_confusion', _conf_matrix)

        df.to_csv(f"{mytag}_dump.csv")

        df[df['true_label'] != df['predicted_label']].apply(
            axis=1,
            func=lambda r: pd.Series({
                'fact': r['text'].split('.')[1],
                'context': r['text'].split('.')[0],
                'type': "False Negative" if r['true_label'] == 1 else "False Positive"
            })
        ).to_csv(
            f'{mytag}_errors.csv'
        )

        return {
            f'{mytag}_loss': _loss_mean,
            f'{mytag}_conf_matrx': _conf_matrix,
            f'{mytag}_f1_score': _f1_score,
            "progress_bar": {
                f"{mytag}_accuracy": val_acc,
                f"{mytag}_f1_score": _f1_score,
            }
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
                          eps=float(self.hparams["adam_epsilon"]))

        return optimizer

    @pl.data_loader
    def train_dataloader(self):
        return DataLoader(
            self.dataloader(pathlib.Path(self.hparams["benchmark_path"]) / 'train.csv'),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate,
            num_workers=self.hparams['cpu_limit']
        )

    @pl.data_loader
    def val_dataloader(self):
        return DataLoader(
            self.dataloader(pathlib.Path(self.hparams["benchmark_path"]) / 'eval.csv'),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate,
            num_workers=self.hparams['cpu_limit']
        )

    @pl.data_loader
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.dataloader(pathlib.Path(self.hparams["benchmark_path"]) / 'test.csv'),
            batch_size=self.hparams["batch_size"], collate_fn=self.collate,
            num_workers=self.hparams['cpu_limit']
        )

    def dataloader(self, x_path: Union[str, pathlib.Path]):
        # TODO: fix this
        df: pd.DataFrame = pd.read_csv(x_path).fillna('')
        df["text"] = df.apply(
            axis=1,
            func=lambda r: "{} </s></s> {}".format(r['question'], r['context'])
        )
        # "id2label": {
        #     "0": "CONTRADICTION",
        #     "1": "NEUTRAL",
        #     "2": "ENTAILMENT"
        # },
        df['label'] = df['label'].apply(lambda l: {0: 0, 1: 2}[int(l)])
        return ClassificationDataset(df[["text", "label"]].to_dict("record"))

    def collate(self, examples):
        batch_size = len(examples)
        df = pd.DataFrame(examples)
        results = self.tokenizer.batch_encode_plus(
            df['text'].values.tolist(),
            add_special_tokens=True,
            max_length=self.hparams["max_length"],
            return_tensors='pt',
            return_token_type_ids=True,
            # return_attention_masks=True,
            pad_to_max_length=True,
            truncation=True,
        )

        assert results["input_ids"].shape[0] == batch_size, \
            f"Invalid shapes {results['input_ids'].shape} {batch_size}"

        return {
            **results,
            "labels": torch.LongTensor(df["label"]),
            "text": df['text'].values,
        }


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModule(config)

    # loader = _module.train_dataloader()
    # batch = next(iter(loader))
    # output = _module.forward(batch)
    # IPython.embed()
    # exit()
    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        show_progress_bar=True,
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config.warmup_steps,
        resume_from_checkpoint=None,
    )

    if config.do_train is True:
        trainer.fit(_module)

    trainer.test(_module)


if __name__ == '__main__':
    main()