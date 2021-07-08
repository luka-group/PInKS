import functools
import itertools
import logging
import pathlib
import random
from functools import partial
from typing import *

import IPython
import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import datasets
# from datasets import load_dataset

from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification

from ModifiedLMData import LMDataModule, DataCollatorForPreconditionWordMask
from ModifiedLangModeling import ModifiedLMModule
from Utils import ClassificationDataset, config_to_hparams

logger = logging.getLogger(__name__)


class NLIMaskedModule(ModifiedLMModule):

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
        # df = pd.DataFrame.from_dict({
        #     'predicted_label': torch.argmax(_logits, dim=1).detach().cpu().numpy(),
        #     'true_label': _labels.detach().cpu().numpy(),
        # }, orient='columns')
        # df['text'] = [s for o in outputs for s in o[f"{mytag}_batch_text"]]
        # df['loss'] = list(
        #     itertools.chain.from_iterable(itertools.repeat(x, batch_size) for x in _loss)
        # )[:len(df)]
        # df['predicate'] = df['text'].apply(self._get_predicate_from_text)
        #
        # _f1_score = f1_score(y_true=df['true_label'], y_pred=df['predicted_label'], average='micro')
        #
        # per_predicate_results = self._compute_metrics(df, mytag, 'All')
        # for pred, gdf in df.groupby('predicate'):
        #     per_predicate_results.update(self._compute_metrics(gdf, mytag, pred))
        #
        # return {
        #     **per_predicate_results,
        #     "progress_bar": {
        #         f"{mytag}_accuracy": per_predicate_results[f'{mytag}_All_accuracy'],
        #         f"{mytag}_f1_score": per_predicate_results[f'{mytag}_All_f1_score'],
        #     }
        # }

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
        self.logger.experiment.add_scalar(f'{self.extra_tag}_total_{spl_name}_{predicate}_loss', _loss_mean)
        self.logger.experiment.add_scalar(f'{self.extra_tag}_total_{spl_name}_{predicate}_acc', _val_acc)
        self.logger.experiment.add_scalar(f'{self.extra_tag}_{spl_name}_{predicate}_f1_macro', _f1_score)

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


class NLIMaskedData(LMDataModule):
    @staticmethod
    def _get_preprocess_func_4_model() -> Callable[[str, str], str]:
        # model_name = self.config['model']
        # template = {
        #     'roberta': '{question} {conj} {context}',
        #     'bart': '{question} {conj} {context}',
        #     'deberta': '[CLS] {question} {conj} {context} [SEP]'
        # }

        pos_conj = ['only if', 'subject to', 'in case', 'contingent upon', 'given', 'if', 'in the case that',
                    "in case", "in the case that", "in the event", "on condition", "on the assumption",
                    "on these terms", "subject to", "supposing", "with the proviso"]

        neg_conj = ["except for", "but", "except", "excepting that", "if not", "lest", "saving", "without", "unless"]

        return partial(
            lambda pc, nc, act, prec, l: '{action} {conj} {precondition}'.format(
                action=act.strip(), precondition=prec.strip(),
                conj={
                    # negative
                    # 0: random.choice(nc),
                    # 'CONTRADICT': random.choice(nc),
                    # # positive
                    # 2: random.choice(pc),
                    # 'ENTAILMENT': random.choice(pc),
                    0: nc[0],
                    'CONTRADICT': nc[0],
                    # positive
                    2: pc[0],
                    'ENTAILMENT': pc[0],
                }[l]
            ),
            pos_conj, neg_conj
        )

    @staticmethod
    def _tokenize_function(_examples: Dict[str, List], _tokenizer, _to_text):
        df = pd.DataFrame.from_dict(_examples).fillna('')
        sents = df.apply(axis=1, func=lambda r: _to_text(r['action'], r['precondition'], r['label'])).values.tolist()

        # logger.info(f"{sents}")
        return {**_tokenizer(sents,), "unmasked_text": sents, "nli_label": df['label'].values.tolist()}

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_setup.model_name,
            cache_dir="/nas/home/qasemi/model_cache",
        )

        all_datasets = datasets.DatasetDict({
            'train': datasets.load_dataset(
                'csv', data_files={"train": self.config.weak_cq_path}
            )['train'],
            'test': datasets.load_dataset(
                'csv', data_files={"test": self.config.cq_path}
            )['test'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
        })

        columns_names = all_datasets.column_names
        _prep_func = self._get_preprocess_func_4_model()

        all_tokenized = all_datasets.map(
            functools.partial(
                self._tokenize_function,
                _tokenizer=functools.partial(
                    tokenizer,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data_module.max_seq_length,
                    return_special_tokens_mask=True,
                    return_tensors='np',
                    return_token_type_ids=True,
                    # add_special_tokens=True,
                ),
                _to_text=_prep_func
            ),
            batched=True,
            batch_size=self.config.data_module.preprocessing_batch_size,
            num_proc=self.config.data_module.preprocessing_num_workers,
            # remove_columns=columns_names,
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

        self.data_collator = ModifiedDataCollatorForPreconditionWordMask(
            tokenizer=tokenizer,
            # mlm_probability=self.config.data_module.mlm_probability
        )

        self.train_dataset = all_tokenized['train'].remove_columns(columns_names['train'])
        self.test_dataset = all_tokenized['test'].remove_columns(columns_names['test'])


class ModifiedDataCollatorForPreconditionWordMask(DataCollatorForPreconditionWordMask):
    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) \
            -> Dict[str, torch.Tensor]:
        # d = LMDataModule.__call__(self, examples)
        # IPython.embed()
        d = super(DataCollatorForPreconditionWordMask, self).__call__(examples)
        # exit()
        return d


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # ------------
    # data
    # ------------
    data_module = NLIMaskedData(config)

    # ------------
    # model
    # ------------
    lmmodel = NLIMaskedModule(config)

    # trainer = pl.Trainer(fast_dev_run=1)
    # data_module.setup('test')
    # loader = data_module.test_dataloader()
    #
    # outputs = []
    # for i, batch in zip(range(4), loader):
    #     outputs.append(
    #         lmmodel.test_step(batch, i)
    #     )
    # IPython.embed()
    # exit()
    # out = lmmodel.test_epoch_end(outputs)

    # lmmodel.load_from_checkpoint('~/CQplus/Outputs/ModifiedLangModeling/lightning_logs/version_1/checkpoints/epoch=0-step=28305.ckpt')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        # gpus='',
        max_epochs=1,
        min_epochs=1,
        resume_from_checkpoint=None,
        distributed_backend=None,
        accumulate_grad_batches=config['trainer_args']['accumulate_grad_batches'],
        limit_train_batches=config['trainer_args']['limit_train_batches'],
    )

    trainer.fit(lmmodel, datamodule=data_module)
    trainer.save_checkpoint(f'Checkpoint/{lmmodel.__class__.__name__}.ckpt')
    trainer.test(lmmodel, datamodule=data_module)


if __name__ == '__main__':
    main()
