import functools
import pathlib
from typing import Callable, Dict, Any, Optional, Union, List

import IPython
import datasets
import numpy as np
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.utils import class_weight

import logging
logger = logging.getLogger(__name__)


class NLIDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()

        assert 'train_composition' in config.data_module, f'Invalid: {config.data_module}'
        self.config: omegaconf.dictconfig.DictConfig = config

        self.train_dataset: Optional[datasets.Dataset] = None
        self.eval_dataset: Optional[datasets.Dataset] = None
        self.test_dataset: Optional[datasets.Dataset] = None

        self.data_collator: Optional[transformers.DataCollator] = None
        self.class_weight: Optional[Dict[int, float]] = None

        logger.warning(f'Remove old class_weights.csv')
        if pathlib.Path('class_weights.csv').exists():
            pathlib.Path('class_weights.csv').unlink()

    def _get_preprocess_func_4_model(self) -> Callable[[str, str], str]:
        model_name = self.config.model_setup.model_name
        template = {
            'roberta': '{action} </s></s> {precondition}',
            'bart': '{action} </s></s> {precondition}',
            'deberta': '[CLS] {action} [SEP] {precondition} [SEP]'
        }
        for k, temp in template.items():
            if k in model_name:
                return functools.partial(lambda t, q, c: t.format(action=q, precondition=c), temp)
        raise ValueError(f'Invalid model name: {model_name}')

    @staticmethod
    def tokenize_function(_examples, _tokenizer: Callable[[str], Dict[str, Any]],
                          _to_text_func: Callable[[str, str], str]) -> Dict[str, Any]:
        df = pd.DataFrame.from_dict(_examples).fillna('')
        sents = df.apply(
            axis=1,
            func=lambda r: _to_text_func(r['action'], r['precondition'])
        ).fillna('').values.tolist()

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

        all_datasets = self._load_all_datasets()

        columns_names = all_datasets.column_names

        _prep_func = self._get_preprocess_func_4_model()
        self.all_tokenized = all_datasets.map(
            functools.partial(
                self.tokenize_function,
                _tokenizer=functools.partial(
                    tokenizer.batch_encode_plus,
                    padding='max_length',
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
            remove_columns=['label', 'action', 'precondition'],
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

        self._update_class_weights()

        self._group_data_in_train_test_dev(columns_names)
        # self.data_collator = transformers.DataCollatorWithPadding(
        #     tokenizer=tokenizer,
        #     padding='max_length',
        #     max_length=self.config.model_setup.max_length,
        # )

    def _update_class_weights(self):
        train_labels = self.all_tokenized['train']['nli_label']
        self.class_weight = dict(zip(
            np.unique(train_labels),
            class_weight.compute_class_weight(
                'balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )
        ))
        for v in [0, 1, 2]:
            if v not in self.class_weight:
                logger.error(f'Did not find value with label {v} in training data')
                self.class_weight[v] = 1.0

        logger.warning(f'Class Weights: {self.class_weight}')
        pd.DataFrame.from_dict(self.class_weight, orient='index').to_csv('class_weights.csv')

    def _load_all_datasets(self):
        func_lut = {
            'mnli': self._load_mnli,
            'dnli': self._load_dnli,
            'weakcq': self._load_weakcq,
            'atomic': self._load_atomic,
        }

        # read the cq data for evaluation and testing, also possible to use the train portion
        cq_dataset = self._load_cq()
        extra_dataset = []
        if 'cq' in self.config.data_module.train_composition:
            extra_dataset = [cq_dataset['cq_train']]

        # real and put together all portions of the dataset.
        all_datasets = datasets.DatasetDict({
            'train': datasets.concatenate_datasets([
                func_lut[name]() for name in self.config.data_module.train_composition if name != 'cq'
            ] + extra_dataset),
            'test': cq_dataset['cq_test'],
            'eval': cq_dataset['cq_valid'],
        })

        return all_datasets

    def _load_mnli(self):
        _dataset = (
            datasets.load_dataset('json', data_files=self.config.mnli_path)['train']
            .shuffle()
            .rename_columns({
                'sentence1': 'action',
                'sentence2': 'precondition',
                'gold_label': 'label',
            })
        )
        return self._trim_size_if_applicable(_dataset, name='mnli')

    def _trim_size_if_applicable(self, _dataset, name: str):
        n_key = 'n_{}_samples'.format(name)
        if n_key in self.config and self.config[n_key] is not None and self.config[n_key] > 0:
            return _dataset.select([i for i in range(int(self.config[n_key]))])
        return _dataset

    def _load_dnli(self):
        _dataset = (
            datasets.load_dataset('csv', data_files=self.config.dnli_path)['train']
            .shuffle()
            .rename_columns({
                'question': 'action',
                'context': 'precondition',
                'label': 'label',
            })
        )

        return self._trim_size_if_applicable(_dataset, name='dnli')

    def _load_weakcq(self):
        _dataset = (
            datasets.load_dataset('csv', data_files=self.config.weak_cq_path)['train']
            .shuffle()
        )

        return self._trim_size_if_applicable(_dataset, name='weakcq')

    def _load_atomic(self):
        _dataset = (
            datasets.load_dataset('csv', data_files=self.config.atomic_nli_path)['train']
            .shuffle()
            .rename_columns({
                'question': 'action',
                'context': 'precondition',
                'label': 'label',
            })
        )

        return self._trim_size_if_applicable(_dataset, name='atomic')

    def _load_cq(self):
        cq_test_path = pathlib.Path(self.config.cq_path)
        cq_eval_path = cq_test_path.parent / cq_test_path.name.replace('test', 'eval')
        cq_train_path = cq_test_path.parent / cq_test_path.name.replace('test', 'train')

        cq_train_dataset = datasets.load_dataset('csv', data_files=str(cq_train_path))['train'].rename_columns({
            'question': 'action',
            'context': 'precondition',
        })

        _dataset = datasets.DatasetDict({
            'cq_train': self._trim_size_if_applicable(cq_train_dataset, name='cq'),
            'cq_valid': datasets.load_dataset('csv', data_files=str(cq_eval_path))['train'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
            'cq_test': datasets.load_dataset('csv', data_files=str(cq_test_path))['train'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
        })
        return _dataset

    def _group_data_in_train_test_dev(self, columns_names):
        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = self.all_tokenized['train'].rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset = self.all_tokenized['test'].rename_columns({
            'nli_label': 'labels'
        })

        self.eval_dataset = self.all_tokenized['eval'].rename_columns({
            'nli_label': 'labels'
        })

        # Not sure if it is userfull
        self.train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
        self.test_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
        self.eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )

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

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.eval_dataset,
            batch_size=self.config.data_module.train_batch_size,
            # collate_fn=self.data_collator,
            num_workers=self.config.data_module.dataloader_num_workers,
        )