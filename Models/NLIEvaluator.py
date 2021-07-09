import functools
import itertools
import logging
import pathlib
from functools import partial
from typing import *

import IPython
import hydra
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import torch
import datasets
import transformers
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, AutoModelForSequenceClassification

from Utils import ClassificationDataset, config_to_hparams

logger = logging.getLogger(__name__)


class NLIModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.hparams = config_to_hparams(config)
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
        })
        loss = results.loss
        # logits = results.logits
        # weighted_loss = self.loss_func(logits, batch["labels"])
        if self.logger is not None:
            self.logger.experiment.add_scalar('train_loss', loss)
            # self.logger.experiment.add_scalar('train_loss', weighted_loss)
        return {
            "loss": loss,
            # "loss": weighted_loss,
            # "text": batch['unmasked_text'],
        }

    def validation_step(self, batch, batch_idx):
        results = self.forward({
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            # 'special_tokens_mask': batch['special_tokens_mask'],
            'token_type_ids': batch['token_type_ids'],
        })
        loss = results.loss
        logits = results.logits

        if self.trainer and self.trainer.use_dp:
            loss = loss.unsqueeze(0)
        return {
            'val_batch_loss': loss.cpu(),
            "val_batch_logits": logits.cpu(),
            "val_batch_labels": batch["nli_label"].cpu(),
            # "val_batch_text": batch['unmasked_text'],
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
        assert len(predicate) == 1, f'Found {predicate} match for {text}'
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
            itertools.chain.from_iterable(itertools.repeat(x, batch_size) for x in _loss)
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

    # def _get_preprocess_func_4_model(self) -> Callable[[str, str], str]:
    #     model_name = self.hparams['model_setup.model_name']
    #     template = {
    #         'roberta': '{question} </s></s> {context}',
    #         'bart': '{question} </s></s> {context}',
    #         'deberta': '[CLS] {question} [SEP] {context} [SEP]'
    #     }
    #     for k, temp in template.items():
    #         if k in model_name:
    #             return partial(lambda t, q, c: t.format(question=q, context=c), temp)
    #     raise ValueError(f'Invalid model name: {model_name}')

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

    # # @pl.data_loader
    # def train_dataloader(self):
    #     logger.info('Loading training data from {}'.format(self.hparams['weak_cq_path']))
    #     df_weak_cq: pd.DataFrame = pd.read_csv(self.hparams['weak_cq_path']).fillna('')
    #     _pred_func = self._get_preprocess_func_4_model()
    #     df_weak_cq["text"] = df_weak_cq.apply(
    #         axis=1,
    #         func=lambda r: _pred_func(r['action'], r['precondition'])
    #     )
    #     # "id2label": {
    #     #     "0": "CONTRADICTION",
    #     #     "1": "NEUTRAL",
    #     #     "2": "ENTAILMENT"
    #     # },
    #     df_weak_cq['label'] = df_weak_cq['label'].apply(lambda l: {'CONTRADICT': 0, 'ENTAILMENT': 2}[l])
    #
    #     df = df_weak_cq
    #
    #     if self.hparams['mnli_path'] is not None and self.hparams['mnli_path'] != '':
    #         df_mnli = pd.read_json(self.hparams['mnli_path'], lines=True)
    #         if len(df_mnli) == 0:
    #             logger.error('Empty MNLI data at {}'.format(self.hparams['mnli_path']))
    #         else:
    #             df_mnli.rename(columns={
    #                 'sentence1': 'action',
    #                 'sentence2': 'precondition',
    #             }, inplace=True)
    #
    #             logger.info(f'mnli columns: {df_mnli.columns}')
    #             df_mnli['label'] = df_mnli['gold_label'].apply(lambda s: {
    #                 'entailment': 2,
    #                 'neutral': 1,
    #                 'contradiction': 0,
    #             }[s.lower()])
    #             df_mnli['text'] = df_mnli.apply(
    #                 axis=1,
    #                 func=lambda r: _pred_func(r['action'], r['precondition'])
    #             )
    #
    #             df = pd.concat([
    #                 df_mnli[["text", "label"]].sample(10000, axis=0),
    #                 df_weak_cq[["text", "label"]],
    #                 df_mnli[["text", "label"]].sample(10000, axis=0),
    #             ])
    #
    #     logger.info(f'Training data shape: {df.shape}')
    #     return DataLoader(
    #         ClassificationDataset(df[["text", "label"]].to_dict("records")),
    #         batch_size=self.hparams['train_setup.batch_size'], collate_fn=self.collate,
    #         num_workers=self.hparams['hardware.cpu_limit']
    #     )
    #
    # def val_dataloader(self):
    #     return itertools.islice(self.test_dataloader(label='Validation (10)'), 10)
    #
    # def test_dataloader(self, label='Testing') -> Union[DataLoader, List[DataLoader]]:
    #     logger.info('Loading {} data from {}'.format(label, self.hparams['cq_path']))
    #     df: pd.DataFrame = pd.read_csv(self.hparams['cq_path']).fillna('')
    #
    #     _pred_func = self._get_preprocess_func_4_model()
    #     df["text"] = df.apply(
    #         axis=1,
    #         func=lambda r: _pred_func(r['question'], r['context'])
    #     )
    #     # "id2label": {
    #     #     "0": "CONTRADICTION",
    #     #     "1": "NEUTRAL",
    #     #     "2": "ENTAILMENT"
    #     # },
    #     df['label'] = df['label'].apply(lambda l: {0: 0, 1: 2, 2: 2}[int(l)])
    #
    #     return DataLoader(
    #         ClassificationDataset(df[["text", "label"]].sample(frac=1).to_dict("records")),
    #         batch_size=self.hparams['train_setup.batch_size'], collate_fn=self.collate,
    #         num_workers=self.hparams['hardware.cpu_limit']
    #     )
    #
    # def collate(self, examples):
    #     batch_size = len(examples)
    #     df = pd.DataFrame(examples)
    #     results = self.tokenizer.batch_encode_plus(
    #         df['text'].values.tolist(),
    #         add_special_tokens=True,
    #         padding='max_length',
    #         max_length=self.hparams["model_setup.max_length"],
    #         return_tensors='pt',
    #         return_token_type_ids=True,
    #         truncation=True,
    #     )
    #
    #     assert results["input_ids"].shape[0] == batch_size, \
    #         f"Invalid shapes {results['input_ids'].shape} {batch_size}"
    #
    #     return {
    #         **results,
    #         "labels": torch.LongTensor(df["label"]),
    #         "text": df['text'].values,
    #     }
    #


class NLIData(pl.LightningDataModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.data_collator = None

    def _get_preprocess_func_4_model(self) -> Callable[[str, str], str]:
        model_name = self.config.model_setup.model_name
        template = {
            'roberta': '{question} </s></s> {context}',
            'bart': '{question} </s></s> {context}',
            'deberta': '[CLS] {question} [SEP] {context} [SEP]'
        }
        for k, temp in template.items():
            if k in model_name:
                return functools.partial(lambda t, q, c: t.format(question=q, context=c), temp)
        raise ValueError(f'Invalid model name: {model_name}')

    @staticmethod
    def tokenize_function(_examples, _tokenizer: Callable[[str], Dict[str, Any]],
                          _to_text_func: Callable[[str, str], str]) -> Dict[str, Any]:
        df = pd.DataFrame.from_dict(_examples).fillna('')
        sents = df.apply(
            axis=1,
            func=lambda r: _to_text_func(r['action'], r['precondition'])
        ).values.tolist()

        labels = df['label'].apply({
            # Weak CQ data
            **{'CONTRADICT': 0, 'ENTAILMENT': 2},
            # MNLI data
            **{
                'entailment': 2,
                'neutral': 1,
                'contradiction': 0,
            },
            # CQ data
            **{0: 0, 1: 2, 2: 2}
        }.get)

        return {**_tokenizer(sents,), "unmasked_text": sents, "nli_label": labels.values.tolist()}

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_setup.model_name,
            cache_dir="/nas/home/qasemi/model_cache",
        )

        # weak_cq_dataset =
        mnli_dataset = (
            datasets.load_dataset('json', data_files=self.config.mnli_path)['train']
            .shuffle()
            .select([i for i in range(20000)])
            .rename_columns({
                'sentence1': 'action',
                'sentence2': 'precondition',
                'gold_label': 'label',
            })
        )

        all_datasets = datasets.DatasetDict({
            'weak_cq': datasets.load_dataset(
                'csv', data_files=self.config.weak_cq_path
            )['train'],
            'mnli': mnli_dataset,
            'cq': datasets.load_dataset(
                'csv', data_files=self.config.cq_path
            )['train'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
        })

        columns_names = all_datasets.column_names

        _prep_func = self._get_preprocess_func_4_model()
        # data_files["validation"] = self.config.data_module.validation_file

        all_tokenized = all_datasets.map(
            functools.partial(
                self.tokenize_function,
                _tokenizer=functools.partial(
                    tokenizer.batch_encode_plus,
                    padding=True,
                    truncation=True,
                    max_length=self.config.data_module.max_seq_length,
                    # return_special_tokens_mask=True,
                    return_tensors='np',
                    return_token_type_ids=True,
                    # add_special_tokens=True,
                ),
                _to_text_func=_prep_func
            ),
            batched=True,
            num_proc=self.config.data_module.preprocessing_num_workers,
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = datasets.concatenate_datasets([
            all_tokenized['weak_cq'].remove_columns(columns_names['weak_cq']),
            all_tokenized['mnli'].remove_columns(columns_names['mnli'])
        ]).rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset = all_tokenized['cq'].remove_columns(columns_names['cq']).rename_columns({
            'nli_label': 'labels'
        })
        # Not sure if it is userfull
        self.train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
        # self.data_collator = transformers.DataCollatorWithPadding(
        #     tokenizer=tokenizer,
        #     padding='max_length',
        #     max_length=self.config.model_setup.max_length,
        # )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data_module.train_batch_size,
            # collate_fn=self.data_collator,
            num_workers=self.config.data_module.dataloader_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.data_module.train_batch_size,
            # collate_fn=self.data_collator,
            num_workers=self.config.data_module.dataloader_num_workers,
        )


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModule(config)

    nli_data_module = NLIData(config)
    nli_data_module.setup('train')
    batch = next(iter(nli_data_module.train_dataloader()))
    output = _module.training_step(batch, 0)
    IPython.embed()
    exit()

    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=str(config['hardware']['gpus']),
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['train_setup']['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config['train_setup']['warmup_steps'],
        resume_from_checkpoint=None,
        distributed_backend=None,
    )

    logger.info(f'Zero shot results')
    _module.extra_tag = 'zero'
    trainer.test(_module)

    logger.info('Tuning')
    _module.extra_tag = 'fit'
    if config['train_setup']['do_train'] is True:
        trainer.fit(_module)

    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    trainer.test(_module)


if __name__ == '__main__':
    main()
