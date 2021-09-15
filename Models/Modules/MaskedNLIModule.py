from typing import List, Any, Dict

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer

from Models.Modules.ModifiedLangModelingModule import ModifiedLMModule

import logging
logger = logging.getLogger(__name__)


class MaskedNLIModule(ModifiedLMModule):

    def forward(self, x):
        return self.model(x).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log('val_loss', out.loss, on_step=True, sync_dist=True)
        return {
            **out,
            **batch
        }

    def test_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log('test_loss', out.loss, on_step=True, sync_dist=True)
        return {
            **out,
            **batch
        }

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        mytag = f'val'
        self._collect_evaluation_results(outputs, mytag)

    def test_epoch_end(self, outputs: List[Any]) -> None:
        mytag = f'test'
        self._collect_evaluation_results(outputs, mytag)

    def _collect_evaluation_results(self, outputs, mytag):
        tokenizer = AutoTokenizer.from_pretrained(
            self.hparams['lm_module.model_name_or_path'],
            cache_dir="/nas/home/qasemi/model_cache",
        )
        _loss = torch.stack([o['loss'] for o in outputs]).detach().numpy()
        _logits = torch.cat([o['logits'] for o in outputs]).detach().numpy()
        _input_ids = torch.cat([o['input_ids'] for o in outputs]).detach().numpy()
        _labels = torch.cat([o['labels'] for o in outputs]).detach().numpy()

        accuracy = ((_labels[_labels != -100]) == _logits.argmax(-1)[_labels != -100]).astype(float).mean()
        logger.info(f'acc={accuracy}')

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

        # terminal logs
        logger.info(f'{self.extra_tag}_{spl_name}_{predicate}_acc={_val_acc}')
        logger.info(f'{self.extra_tag}_{spl_name}_{predicate}_loss={_loss_mean}')
        logger.info(f'{self.extra_tag}_{spl_name}_{predicate}_f1={_f1_score}')
        logger.info(f'{self.extra_tag}_{spl_name}_{predicate}_conf_matrx: \n{_conf_matrix}')

        # PL logs
        self.logger.log_metrics({
            f'{self.extra_tag}_total_{spl_name}_{predicate}_loss': _loss_mean,
            f'{self.extra_tag}_total_{spl_name}_{predicate}_acc': _val_acc,
            f'{self.extra_tag}_{spl_name}_{predicate}_f1_macro': _f1_score,
        })

        # dump csv files
        df_errors.to_csv(f'{self.extra_tag}_{spl_name}_{predicate}_errors.csv')
        df.to_csv(f"{self.extra_tag}_{spl_name}_{predicate}_dump.csv")

        # preparing output
        per_predicate_results = {
            f'{spl_name}_{predicate}_mean_loss': df['loss'].mean(),
            f'{spl_name}_{predicate}_accuracy': _val_acc,
            f'{spl_name}_{predicate}_conf_matrx': _conf_matrix,
            f'{spl_name}_{predicate}_f1_score': _f1_score,
        }
        return per_predicate_results