import itertools
from typing import Dict, List, Any

import IPython
import pandas as pd

import pytorch_lightning as pl
import torch
# from pytorch_lightning.loggers import wandb
import wandb
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW

from Models import Utils

import logging
logger = logging.getLogger(__name__)


class NLIModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = Utils.config_to_hparams(config)
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
        self.extra_tag = ''

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache",
            use_fast=False
        )

        self.embedder = AutoModelForSequenceClassification.from_pretrained(
            self.hparams["model_setup.model_name"],
            cache_dir="/nas/home/qasemi/model_cache"
        )

    def forward(self, batch):

        # additional_params = {}
        # if "roberta" in self.hparams['model_setup.model_name']:
        #     additional_params["token_type_ids"] = None
        # elif "bart" in self.hparams['model_setup.model_name']:
        #     pass
        # else:
        #     additional_params["token_type_ids"] = batch["token_type_ids"]
        # # batch["token_type_ids"] = None if "roberta" in self.hparams['model'] else batch["token_type_ids"]
        #
        # label_dict = {}
        # if 'labels' in batch:
        #     label_dict = {'labels': batch['labels']}

        results = self.embedder(
            **batch
            # input_ids=batch["input_ids"],
            # attention_mask=batch["attention_mask"],
            # **additional_params,
            # **label_dict
        )

        return results

    def test_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        loss_dict = self.validation_step(batch, batch_idx)
        return {
            k.replace('val_', 'test_'): v for k, v in loss_dict.items()
        }

    def training_step(self, batch, batch_idx):
        results = self.forward({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            # 'special_tokens_mask': batch['special_tokens_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels'],
        })
        loss = results.loss
        # logits = results.logits
        # loss = self.loss_func(logits, batch["labels"])
        if self.logger is not None:
            self.logger.log_metrics({'train_loss': loss})
            # self.logger.experiment.add_scalar('train_loss', weighted_loss)
        return {
            "loss": loss,
            # "text": batch['unmasked_text'],
        }

    def validation_step(self, batch, batch_idx):
        results = self.forward({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            # 'special_tokens_mask': batch['special_tokens_mask'],
            'token_type_ids': batch['token_type_ids'],
            'labels': batch['labels'],
        })
        loss = results.loss
        logits = results.logits

        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        # logging for early stopping
        self.log('val_loss', loss)

        return {
            'val_batch_loss': loss.detach().cpu(),
            "val_batch_logits": logits.detach().cpu(),
            "val_batch_labels": batch["labels"].detach().cpu(),
            "val_batch_text": batch['unmasked_text'],
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mytag = f'val'
        self._collect_evaluation_results(outputs, mytag)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        mytag = f'test'
        self._collect_evaluation_results(outputs, mytag)

    def _get_predicate_from_text(self, text: str) -> str:
        mappings = {
            'UsedFor': [
                'is typically used',
                'are typically used',
                'You can typically use',
                'can typically be used',
            ],
            'CapableOf': [
                'is typically capable',
                'are typically capable'
            ],
            'Causes': [
                'typically causes',
                'typically cause',
                'typically causes',
                'typically cause'
            ],
            'CausesDesire': [
                'typically causes desire',
                'typically cause desire'
            ],
            'Desires': [
                'typically desires',
                'typically desire',
                'typically do desire',
                'typically does desire',
                'typically have desire',
                'typically has desire',
            ],
        }
        predicate = set()
        for pred, templs in mappings.items():
            if any([t in text for t in templs]):
                predicate.add(pred)

        if predicate == {'Causes', 'CausesDesire'}:
            predicate = {'CausesDesire'}

        try:
            assert len(predicate) == 1, f'Found {predicate} match for {text}'
        except AssertionError as e:
            IPython.embed()
            raise e
        return predicate.pop()

    def _collect_evaluation_results(self, outputs, mytag):
        _loss = torch.stack([o[f'{mytag}_batch_loss'] for o in outputs])
        _logits = torch.cat([o[f"{mytag}_batch_logits"] for o in outputs])
        _labels = torch.cat([o[f"{mytag}_batch_labels"] for o in outputs])
        batch_size = len(outputs[0][f"{mytag}_batch_logits"])

        df = pd.DataFrame.from_dict({
            'predicted_label': torch.argmax(_logits, dim=1).detach().cpu().numpy(),
            'true_label': _labels.detach().cpu().numpy(),
        }, orient='columns')
        df['text'] = [s for o in outputs for s in o[f"{mytag}_batch_text"]]
        df['loss'] = list(
            itertools.chain.from_iterable(itertools.repeat(x.detach().numpy(), batch_size) for x in _loss)
        )[:len(df)]
        df['predicate'] = df['text'].apply(self._get_predicate_from_text)

        _f1_score = f1_score(y_true=df['true_label'], y_pred=df['predicted_label'], average='micro')

        per_predicate_results = self._compute_metrics(df, mytag, 'All')
        for pred, gdf in df.groupby('predicate'):
            per_predicate_results.update(self._compute_metrics(gdf, mytag, pred))

        return {
            **per_predicate_results,
            "progress_bar": {
                f"{mytag}_accuracy": per_predicate_results[f'{mytag}_All_accuracy'],
                f"{mytag}_f1_score": per_predicate_results[f'{mytag}_All_f1_score'],
            }
        }

    def _compute_metrics(self, df: pd.DataFrame, spl_name: str, predicate: str) -> Dict[str, Any]:
        _val_acc = accuracy_score(df['true_label'], df['predicted_label'])
        _loss_mean = df['loss'].mean()
        _f1_score = f1_score(y_true=df['true_label'], y_pred=df['predicted_label'], average='micro')

        _conf_matrix = pd.DataFrame(
            confusion_matrix(y_true=df['true_label'], y_pred=df['predicted_label'], labels=[0, 1, 2]),
            columns=['C', 'N', 'E'],
            index=['C', 'N', 'E'],
        )
        df_errors = (
            df[df['true_label'] != df['predicted_label']].apply(
                axis=1,
                func=lambda r: pd.Series({
                    'fact': r['text'].split('.')[1],
                    'context': r['text'].split('.')[0],
                    'type': "False Negative" if r['true_label'] == 1 else "False Positive"
                })
            )
        )

        prefix = f'{self.extra_tag}_{spl_name}_{predicate}'

        # terminal logs
        logger.info(f'{prefix}_acc={_val_acc}')
        logger.info(f'{prefix}_loss={_loss_mean}')
        logger.info(f'{prefix}_f1={_f1_score}')
        logger.info(f'{prefix}_conf_matrx: \n{_conf_matrix}')

        # PL logs
        if self.logger is not None:
            self.logger.log_metrics({
                # metrics
                f'{prefix}_loss': _loss_mean,
                f'{prefix}_acc': _val_acc,
                f'{prefix}_f1_macro': _f1_score,
                # confusion matrix
                f"{prefix}_conf_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=df['true_label'].values,
                    preds=df['predicted_label'].values,
                    class_names=['C', 'N', 'E']
                ),
                # dump csv files
                f'{prefix}_dump': self._df_to_wandb_table(dataframe=df),
                f'{prefix}_errors': self._df_to_wandb_table(dataframe=df_errors),
            })
            self.logger.log_metrics({})

        df_errors.to_csv(f'{prefix}_errors.csv')
        df.to_csv(f"{prefix}_dump.csv")

        # preparing output
        per_predicate_results = {
            f'{spl_name}_{predicate}_mean_loss': _loss_mean,
            f'{spl_name}_{predicate}_accuracy': _val_acc,
            # f'{spl_name}_{predicate}_conf_matrx': _conf_matrix,
            f'{spl_name}_{predicate}_f1_score': _f1_score,
        }
        return per_predicate_results

    @staticmethod
    def _df_to_wandb_table(dataframe: pd.DataFrame):
        return wandb.Table(
            dataframe=dataframe.applymap(str)
        )

    def configure_optimizers(self):
        model = self.embedder
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams['train_setup.weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=float(self.hparams['train_setup.learning_rate']),
                          eps=float(self.hparams['train_setup.adam_epsilon']),
                          betas=(
                              self.hparams['train_setup.beta1'],
                              self.hparams['train_setup.beta2']
                            ),
                          )

        # optimizer = AdamW(self.parameters(), lr=float(self.hparams["learning_rate"]),
        #                   eps=float(self.hparams["adam_epsilon"]))

        return optimizer