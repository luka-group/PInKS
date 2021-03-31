import itertools
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification
from sklearn.metrics import confusion_matrix, f1_score

import logging

# from LMBenchmarkEvaluator import BaseEvaluationModule
from transformers.modeling_outputs import SequenceClassifierOutput

from Utils import ClassificationDataset

logger = logging.getLogger(__name__)


class NLIModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = dict(config)
        self.root_path = pathlib.Path(__file__).parent.absolute()

        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams['model_setup']['model_name'],
                                                       cache_dir="/nas/home/qasemi/model_cache",
                                                       use_fast=False)

        np_counts = np.array(self.hparams['data_stats']['counts'])
        loss_weights = 1.0-(np_counts/np_counts.sum())
        logger.info(f'Using weights ({loss_weights}) for the loss function.')
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1, reduction="mean",
                                             weight=torch.from_numpy(loss_weights).float())

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            self.hparams['model_setup']['model_name'],
            cache_dir="/nas/home/qasemi/model_cache"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> SequenceClassifierOutput:
        batch["token_type_ids"] = None if "roberta" in self.hparams['model_setup']['model_name'] else batch["token_type_ids"]

        label_dict = {}
        if 'labels' in batch:
            label_dict = {'labels': batch['labels']}

        results = self.embedder(input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                token_type_ids=batch["token_type_ids"],
                                **label_dict)

        return results

    def training_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = results.loss
        logits = results.logits
        weighted_loss = self.loss_func(logits, batch["labels"])
        if self.logger is not None:
            # self.logger.experiment.add_scalar('train_loss', loss)
            self.logger.experiment.add_scalar('train_loss', weighted_loss)
        return {
            # "loss": loss,
            "loss": weighted_loss,
            "text": batch['text'],
        }

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        loss_dict = self.validation_step(batch, batch_idx)
        return {
            k.replace('val_', 'test_'): v for k, v in loss_dict.items()
        }

    def validation_step(self, batch, batch_idx):
        results = self.forward(batch)
        loss = results.loss
        logits = results.logits

        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            'val_batch_loss': loss.cpu(),
            "val_batch_logits": logits.cpu(),
            "val_batch_labels": batch["labels"].cpu(),
            "val_batch_text": batch['text'],
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mytag = 'val'
        self._collect_evaluation_results(outputs, mytag)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        mytag = 'test'
        self._collect_evaluation_results(outputs, mytag)

    def _collect_evaluation_results(self, outputs, mytag):
        _loss_mean = torch.stack([o[f'{mytag}_batch_loss'] for o in outputs]).mean()
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
        df['text'] = [s for o in outputs for s in o[f'{mytag}_batch_text']]

        _f1_score = f1_score(y_true=df['true_label'], y_pred=df['predicted_label'], average='micro')
        self.logger.experiment.add_scalar(f'{mytag}_f1_macro', _f1_score)

        # IPython.embed()

        _conf_matrix = pd.DataFrame(
            confusion_matrix(y_true=df['true_label'], y_pred=df['predicted_label'], labels=[0, 1, 2]),
            columns=['C', 'N', 'E'],
            index=['C', 'N', 'E'],
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
        optimizer = AdamW(self.parameters(), lr=float(self.hparams['train_setup']["learning_rate"]),
                          eps=float(self.hparams['train_setup']["adam_epsilon"]))

        return optimizer

    def train_dataloader(self):
        logger.info('Loading training data from {}'.format(self.hparams['weak_cq_path']))
        df: pd.DataFrame = pd.read_csv(self.hparams['weak_cq_path']).fillna('')
        df["text"] = df.apply(
            axis=1,
            func=lambda r: "{} </s></s> {}".format(r['action'], r['precondition'])
        )
        # "id2label": {
        #     "0": "CONTRADICTION",
        #     "1": "NEUTRAL",
        #     "2": "ENTAILMENT"
        # },
        df['label'] = df['label'].apply(lambda l: {'CONTRADICT': 0, 'ENTAILMENT': 2}[l])
        return DataLoader(
            ClassificationDataset(df[["text", "label"]].to_dict("records")),
            batch_size=self.hparams['train_setup']['batch_size'], collate_fn=self.collate,
            num_workers=self.hparams['hardware']['cpu_limit']
        )

    def val_dataloader(self):
        return itertools.islice(self.test_dataloader(label='Validation (10)'), 10)

    def test_dataloader(self, label='Testing') -> Union[DataLoader, List[DataLoader]]:
        logger.info('Loading {} data from {}'.format(label, self.hparams['cq_path']))
        df: pd.DataFrame = pd.read_csv(self.hparams['cq_path']).fillna('')
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

        return DataLoader(
            ClassificationDataset(df[["text", "label"]].sample(frac=1).to_dict("records")),
            batch_size=self.hparams['train_setup']['batch_size'], collate_fn=self.collate,
            num_workers=self.hparams['hardware']['cpu_limit']
        )

    def collate(self, examples):
        batch_size = len(examples)
        df = pd.DataFrame(examples)
        results = self.tokenizer.batch_encode_plus(
            df['text'].values.tolist(),
            add_special_tokens=True,
            max_length=self.hparams['model_setup']['max_length'],
            return_tensors='pt',
            return_token_type_ids=True,
            # return_attention_masks=True,
            # pad_to_max_length=True,
            truncation=True,
            padding=True,
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
    # IPython.embed()
    # exit()
    # batch = next(iter(loader))
    # IPython.embed()
    # exit()
    # output = _module.forward(batch)
    # output = _module.training_step(batch, 0)

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