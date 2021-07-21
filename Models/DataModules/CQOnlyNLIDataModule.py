import functools
import pathlib
from typing import Callable, Dict, Any, Optional

import datasets
import omegaconf
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from Models.DataModules.BaseNLIDataModule import BaseNLIDataModule


class CQOnlyNLIDataModule(BaseNLIDataModule):

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_setup.model_name,
            cache_dir="/nas/home/qasemi/model_cache",
        )
        test_path = pathlib.Path(self.config.cq_path)
        train_path = test_path.parent / test_path.name.replace('test', 'train')
        all_datasets = datasets.DatasetDict({
            'train': datasets.load_dataset(
                'csv', data_files=train_path
            )['train'].shuffle().select([i for i in range(500)]),
            'test': datasets.load_dataset(
                'csv', data_files=test_path
            )['train'],
        }).rename_columns({
            'question': 'action',
            'context': 'precondition',
        })

        columns_names = all_datasets.column_names

        _prep_func = self._get_preprocess_func_4_model()
        # data_files["validation"] = self.config.data_module.validation_file

        self.all_tokenized = all_datasets.map(
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

        self._group_data_in_train_test_dev(columns_names)
