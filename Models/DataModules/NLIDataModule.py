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

        self.all_tokenized: Optional[datasets.Dataset] = None

        logger.warning(f'Remove old class_weights.csv')
        if pathlib.Path('class_weights.csv').exists():
            pathlib.Path('class_weights.csv').unlink()

    def _get_preprocess_func_4_model(self) -> Callable[[str, str], str]:
        model_name = self.config.model_setup.model_name
        template = {
            'roberta': '{hypothesis} </s></s> {premise}',
            'bart': '{hypothesis} </s></s> {premise}',
            'deberta': '[CLS] {hypothesis} [SEP] {premise} [SEP]'
        }
        for k, temp in template.items():
            if k in model_name:
                return functools.partial(lambda t, q, c: t.format(hypothesis=q, premise=c), temp)
        raise ValueError(f'Invalid model name: {model_name}')

    @staticmethod
    def tokenize_function(_examples, _tokenizer: Callable[[str], Dict[str, Any]],
                          _to_text_func: Callable[[str, str], str]) -> Dict[str, Any]:
        df = pd.DataFrame.from_dict(_examples).fillna('')
        sents = df.apply(
            axis=1,
            func=lambda r: _to_text_func(r['hypothesis'], r['premise'])
        ).fillna('').values.tolist()

       #  labels = df['label'].apply({
       #     # Weak CQ data
       #     **{'CONTRADICT': 0, 'ENTAILMENT': 2},
       #     # MNLI data
       #     **{
       #         'entailment': 2,
       #         'neutral': 1,
       #         'contradiction': 0,
       #     },
       #     # CQ data
       #     **{0: 0, 1: 2, 2: 2}
       # }.get)

        return {**_tokenizer(sents, ), "unmasked_text": sents, "labels": df['label'].values.tolist()}

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_setup.model_name,
            cache_dir="/nas/home/qasemi/model_cache",
        )

        all_datasets = self._load_all_datasets()

        columns_names = all_datasets.column_names

        _prep_func = self._get_preprocess_func_4_model()

        logger.info(f'Tokenizing dataset')
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
            remove_columns=['label', 'hypothesis', 'premise'],
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

        logger.info(f'Grouping dataset')
        self._update_class_weights()

        self._group_data_in_train_test_dev(columns_names)

    def _update_class_weights(self):
        train_labels = self.all_tokenized['train']['integer_label']
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
            'cq': self._load_cq,
            'winoventi': self._load_winoventi,
            'anion': self._load_anion,
        }

        assert set(self.config.data_module.train_composition).intersection(
            set(self.config.data_module.test_composition)).difference(
            {'cq', 'dnli'}) == set(), f'INVALID train/test composition'

        all_loaded = {dname: func_lut[dname]() for dname in set(
            self.config.data_module.train_composition).union(
            set(self.config.data_module.test_composition)
        )}

        logger.info(f'Loading all the datasets')
        all_datasets = datasets.DatasetDict({
            'train': datasets.concatenate_datasets(
                [all_loaded[name]['train'] for name in self.config.data_module.train_composition]
            ),
            'test': datasets.concatenate_datasets(
                [all_loaded[name]['test'] for name in self.config.data_module.test_composition]
            ),
            'eval': datasets.concatenate_datasets(
                [all_loaded[name]['eval'] for name in self.config.data_module.test_composition
                 if 'eval' in all_loaded[name]]
            ),
        })

        return all_datasets

    def _load_mnli(self):
        _dataset =  datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('json', data_files=self.config.mnli_path)['train'].shuffle().rename_columns({
                        # 'sentence1': 'action',
                        'sentence1': 'hypothesis',
                        # 'sentence2': 'precondition',
                        'sentence2': 'premise',
                        'gold_label': 'label',
                    }),
                    name='mnli'
                ),
                d = {
                    'entailment': 2,
                    'neutral': 1,
                    'contradiction': 0,
                }
            )
        })
        return self._filter_extra_columns(_dataset)

    @staticmethod
    def _translate_labels(_dataset, d:Dict = {}):
        return _dataset.map(
            function=lambda r: {
                'hypothesis': r['hypothesis'],
                'premise': r['premise'],
                'label': d[r['label']],
        })

    def _trim_size_if_applicable(self, _dataset, name: str):
        n_key = 'n_{}_samples'.format(name)

        if n_key in self.config and self.config[n_key] is not None and self.config[n_key] > 0:
            return _dataset.select([i for i in range(min(int(self.config[n_key]), len(_dataset)))])

        # FIXME there is problem here when we do not run the select and skip this if.
        #  Not sure why but the system throws keyMismatch error during the concatenation. As if like the rename
        #  column has not worked at all
        return self._fix_dataset_bug(_dataset)

    @staticmethod
    def _fix_dataset_bug(_dataset):
        return _dataset.select([i for i in range(int(len(_dataset)))])

    def _load_dnli(self):
        dnli_test_path = pathlib.Path(self.config.dnli_path)
        dnli_eval_path = dnli_test_path.parent / dnli_test_path.name.replace('test', 'dev')
        dnli_train_path = dnli_test_path.parent / dnli_test_path.name.replace('test', 'train')

        column_lut = {
            'question': 'hypothesis',
            'context': 'premise',
            'label': 'label',
        }

        _dataset = datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('csv', data_files=str(dnli_train_path))['train'].shuffle(),
                    name='dnli'
                ).rename_columns(column_lut),
                d={0: 0, 1: 2, 2: 2}
            ),
            'eval': self._translate_labels(
                self._fix_dataset_bug(
                    datasets.load_dataset('csv', data_files=str(dnli_eval_path))['train'].rename_columns(column_lut)
                ),
                d={0: 0, 1: 2, 2: 2}
            ),
            'test': self._translate_labels(
                self._fix_dataset_bug(
                    datasets.load_dataset('csv', data_files=str(dnli_test_path))['train'].rename_columns(column_lut)
                ),
                d={0: 0, 1: 2, 2: 2}
            ),
        })

        return self._filter_extra_columns(_dataset)

    def _load_winoventi(self):
        logger.info(f'Loading Winoventi')
        return datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('csv', data_files=self.config.winoventi_nli_path)['train']
                    .shuffle(),
                    name='winoventi'
                ),
                d = {0: 0, 1: 2, 2: 2}
            )
        })

    def _load_anion(self):
        logger.info(f'Loading ANION')
        anion_test = str(pathlib.Path(self.config.anion_nli_path))
        anion_train = str(pathlib.Path(self.config.anion_nli_path)).replace('test', 'train')
        anion_eval = str(pathlib.Path(self.config.anion_nli_path)).replace('test', 'dev')

        return datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('csv', data_files=anion_train)['train']
                    .shuffle(),
                    name='anion'
                ),
                d = {0: 0, 1: 2, 2: 2}
            ),
            'eval': self._translate_labels(
                self._fix_dataset_bug(datasets.load_dataset('csv', data_files=anion_eval)['train']),
                d = {0: 0, 1: 2, 2: 2}
            ),
            'test': self._translate_labels(
                self._fix_dataset_bug(datasets.load_dataset('csv', data_files=anion_test)['train']),
                d={0: 0, 1: 2, 2: 2}
            ),
        })

    def _load_weakcq(self):
        _dataset = (
            datasets.load_dataset('csv', data_files=self.config.weak_cq_path)['train']
            .shuffle()
            .rename_columns({
                'action': 'hypothesis',
                'precondition': 'premise',
                'label': 'label',
            })
        )

        old_len = len(_dataset)
        if 'weakcq_recal_threshold' in self.config:
            def check_recal(r, lim):
                return r['recall'] >= lim
            _dataset = _dataset.filter(
                function=functools.partial(check_recal, lim=float(self.config.weakcq_recal_threshold)),
                num_proc=self.config.data_module.preprocessing_num_workers,
                load_from_cache_file=not self.config.data_module.overwrite_cache,
            )
            logger.info(f'Filter weakCQ on threshold {self.config.weakcq_recal_threshold}, '
                        f'old_len:{old_len}, new_len:{len(_dataset)}')

        return self._filter_extra_columns(datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(_dataset, name='weakcq'),
                d={
                    0: 0, 1: 2, 2: 2,
                    'CONTRADICT': 0, 'ENTAILMENT': 2
                }
            )
        }))

    def _load_atomic(self):
        return datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('csv', data_files=self.config.atomic_nli_path)['train']
                    .shuffle()
                    .rename_columns({
                        'question': 'hypothesis',
                        'context': 'premise',
                        'label': 'label',
                    }),
                    name='atomic'
                ),
                d={0: 0, 1: 2, 2: 2}
            )
        })

    def _load_cq(self):
        cq_test_path = pathlib.Path(self.config.cq_path)
        cq_eval_path = cq_test_path.parent / cq_test_path.name.replace('test', 'eval')
        cq_train_path = cq_test_path.parent / cq_test_path.name.replace('test', 'train')

        column_lut = {
            'question': 'hypothesis',
            'context': 'premise',
            'label': 'label'
        }
        _dataset = datasets.DatasetDict({
            'train': self._translate_labels(
                self._trim_size_if_applicable(
                    datasets.load_dataset('csv', data_files=str(cq_train_path))['train'].rename_columns(column_lut),
                    name='cq'
                ),
                d={0: 0, 1: 2, 2: 2},
            ),
            'eval': self._translate_labels(
                self._fix_dataset_bug(
                    datasets.load_dataset('csv', data_files=str(cq_eval_path))['train'].rename_columns(column_lut)
                ),
                d={0: 0, 1: 2, 2: 2},
            ),
            'test': self._translate_labels(
                self._fix_dataset_bug(
                    datasets.load_dataset('csv', data_files=str(cq_test_path))['train'].rename_columns(column_lut)
                ),
                d={0: 0, 1: 2, 2: 2},
            ),
        })
        return self._filter_extra_columns(_dataset)

    def _filter_extra_columns(self, _dataset):
        return _dataset.remove_columns(
            column_names=set(_dataset['train'].features.keys()).difference({'premise', 'hypothesis', 'label'})
        )

    def _group_data_in_train_test_dev(self, columns_names):
        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = self.all_tokenized['train']
        self.test_dataset = self.all_tokenized['test']
        self.eval_dataset = self.all_tokenized['eval']

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
