import functools
import pathlib
from typing import Callable, Dict, Any, Optional, Union, List

import IPython
import datasets
import omegaconf
import pandas as pd
import pytorch_lightning as pl
import transformers
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class BaseNLIDataModule(pl.LightningDataModule):
    def prepare_data(self):
        pass

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config: omegaconf.dictconfig.DictConfig = config
        self.train_dataset: Optional[datasets.Dataset] = None
        self.eval_dataset: Optional[datasets.Dataset] = None
        self.test_dataset: Optional[datasets.Dataset] = None
        self.data_collator: Optional[transformers.DataCollator] = None
        self.all_tokenized: Optional[datasets.DatasetDict] = None

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

        # weak_cq_dataset =
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
            load_from_cache_file=not self.config.data_module.overwrite_cache,
        )

        self._group_data_in_train_test_dev(columns_names)
        # self.data_collator = transformers.DataCollatorWithPadding(
        #     tokenizer=tokenizer,
        #     padding='max_length',
        #     max_length=self.config.model_setup.max_length,
        # )

    def _load_all_datasets(self):
        mnli_dataset = (
            datasets.load_dataset('json', data_files=self.config.mnli_path)['train']
            .shuffle()
            .rename_columns({
                'sentence1': 'action',
                'sentence2': 'precondition',
                'gold_label': 'label',
            })
        )
        if 'n_MNLI_samples' in self.config and self.config.n_MNLI_samples is not None:
            mnli_dataset = mnli_dataset.select([i for i in range(int(self.config.n_MNLI_samples))])

        weakcq_dataset = (
            datasets.load_dataset('csv', data_files=self.config.weak_cq_path)['train']
            .shuffle()
        )
        if 'n_WeakCQ_samples' in self.config and self.config.n_WeakCQ_samples is not None:
            weakcq_dataset = weakcq_dataset.select([i for i in range(int(self.config.n_WeakCQ_samples))])

            # df_weakcq = weakcq_dataset.to_pandas()
            # df_weakcq['label']
            # IPython.embed()
            # exit()

        cq_path = pathlib.Path(self.config.cq_path)
        cq_train_path = cq_path.parent / cq_path.name.replace('test', 'train')
        all_datasets = datasets.DatasetDict({
            'weak_cq': weakcq_dataset,
            'mnli': mnli_dataset,
            'cq': datasets.load_dataset(
                'csv', data_files=self.config.cq_path
            )['train'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
            'cq_test': datasets.load_dataset(
                'csv', data_files=str(cq_train_path)
            )['train'].rename_columns({
                'question': 'action',
                'context': 'precondition',
            })
        })
        return all_datasets

    def _group_data_in_train_test_dev(self, columns_names):
        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = datasets.concatenate_datasets([
            self.all_tokenized['weak_cq'].remove_columns(columns_names['weak_cq']),
            self.all_tokenized['mnli'].remove_columns(columns_names['mnli'])
        ]).rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset = self.all_tokenized['cq'].remove_columns(columns_names['cq']).rename_columns({
            'nli_label': 'labels'
        })

        self.eval_dataset = self.all_tokenized['cq_test'].remove_columns(columns_names['cq_test']).rename_columns({
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


class MnliTuneCqTestDataModule(BaseNLIDataModule):
    def _group_data_in_train_test_dev(self, columns_names):
        self.train_dataset = self.all_tokenized['mnli'].remove_columns(columns_names['mnli']).rename_columns({
            'nli_label': 'labels'
        })
        self.train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )

        self.test_dataset = self.all_tokenized['cq'].remove_columns(columns_names['cq']).rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )

        self.eval_dataset = self.all_tokenized['cq_test'].remove_columns(columns_names['cq_test']).rename_columns({
            'nli_label': 'labels'
        })
        self.eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )


class WeakTuneCqTestDataModule(BaseNLIDataModule):
    def _group_data_in_train_test_dev(self, columns_names):
        self.train_dataset = self.all_tokenized['weak_cq'].remove_columns(columns_names['weak_cq']).rename_columns({
            'nli_label': 'labels'
        })
        self.train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )

        self.test_dataset = self.all_tokenized['cq'].remove_columns(columns_names['cq']).rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )

        self.eval_dataset = self.all_tokenized['cq_test'].remove_columns(columns_names['cq_test']).rename_columns({
            'nli_label': 'labels'
        })
        self.eval_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
