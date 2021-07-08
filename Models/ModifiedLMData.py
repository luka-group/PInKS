import functools
import random

import IPython
import hydra
from typing import *
import omegaconf

import warnings
from datasets import load_dataset
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from transformers import (
    AutoTokenizer,
    DataCollatorForWholeWordMask,
    # DataCollatorForLanguageModeling,
)

import logging
logger = logging.getLogger(__name__)


class LMDataModule(pl.LightningDataModule):
    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        super().__init__()
        self.config = config
        self.train_dataset = None
        self.eval_dataset = None
        self.test_dataset = None
        self.data_collator = None

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.lm_module.model_name_or_path,
            cache_dir="/nas/home/qasemi/model_cache",
        )
        extension = self.config.data_module.train_file.split(".")[-1]
        if extension in ("txt", "raw"):
            extension = "text"

        data_files = {}
        data_files["train"] = self.config.data_module.train_file
        # data_files["validation"] = self.config.data_module.validation_file
        datasets = load_dataset(extension, data_files=data_files, **dict(self.config.data_module.datasetloader_kwargs))

        column_names = datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if self.config.data_module.line_by_line:
            # When using line_by_line, we just tokenize each nonempty line.
            padding = "max_length" if self.config.data_module.pad_to_max_length else False

            def tokenize_function(_examples, _tokenizer):
                # Remove empty lines
                _examples["text"] = [line for line in _examples["text"]
                                     if len(line) > 0 and not line.isspace()]
                return _tokenizer(
                    _examples["text"],
                )

            tokenized_datasets = datasets.map(
                functools.partial(
                    tokenize_function,
                    _tokenizer=functools.partial(
                        tokenizer,
                        padding=True,
                        truncation=True,
                        max_length=self.config.data_module.max_seq_length,
                        return_special_tokens_mask=True,
                        return_tensors='np',
                        return_token_type_ids=True,
                        # add_special_tokens=True,
                    )
                ),
                batched=True,
                num_proc=self.config.data_module.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not self.config.data_module.overwrite_cache,
            )

        data_collator = DataCollatorForPreconditionWordMask(
            tokenizer=tokenizer,
            # mlm_probability=self.config.data_module.mlm_probability
        )

        train_dataset = tokenized_datasets["train"]
        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = train_dataset
        # self.eval_dataset = eval_dataset
        self.data_collator = data_collator

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data_module.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.data_module.dataloader_num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.data_module.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.config.data_module.dataloader_num_workers,
        )


class DataCollatorForPreconditionWordMask(DataCollatorForWholeWordMask):

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, tokenizer=tokenizer, **kwargs)

        self.model_name = tokenizer.name_or_path
        self.special_tokens = [
            tokenizer.eos_token,
            tokenizer.unk_token,
            tokenizer.pad_token,
            tokenizer.bos_token,
        ]

        self.mask_word_list = [
            c.split(' ')
            for c in [
                'unless',
                'only if',
                'so',
                'hence',
                'consequently',
                'thus',
                'therefore',
                'as a result',
                'thus',
                'accordingly',
                'because of that',
                'as a consequence',
                'as a result'
                # more enablings
                'only if', 'subject to', 'in case', 'contingent upon', 'given', 'if', 'in the case that',
                "in case", "in the case that", "in the event", "on condition", "on the assumption",
                "on these terms", "subject to", "supposing", "with the proviso",
                # more disablings
                "but", "except", "except for", "excepting that", "if not", "lest", "saving", "without", "unless",
            ]
        ]

        self._is_sub_word = self._gen_is_sub_word(tokenizer.name_or_path)
        self._clean_token = self._gen_clean_token(tokenizer.name_or_path)

    @staticmethod
    def _roberta_gen_is_sub_word(s):
        return u'\u0120' not in s

    @staticmethod
    def _bert_gen_is_sub_word(s):
        return s.startswith("##")

    @staticmethod
    def _gen_is_sub_word(model_name) -> Callable[[str], bool]:
        if 'roberta' in model_name:
            return DataCollatorForPreconditionWordMask._roberta_gen_is_sub_word
        if 'bert' in model_name:
            return DataCollatorForPreconditionWordMask.bert_gen_is_sub_word

    @staticmethod
    def _roberta_gen_clean_token(s):
        return s.replace(u'\u0120', '')

    @staticmethod
    def _bert_gen_clean_token(s):
        return s.replace('##', '')

    @staticmethod
    def _gen_clean_token(model_name) -> Callable[[str], str]:
        if 'roberta' in model_name:
            return DataCollatorForPreconditionWordMask._roberta_gen_clean_token
        if 'bert' in model_name:
            return DataCollatorForPreconditionWordMask._bert_gen_clean_token

    def _whole_word_mask(self, input_tokens: List[str], max_predictions=512):
        """
        Get 0/1 labels for masked tokens with whole word mask proxy
        """
        # logger.info(f'input_tokens: {input_tokens}')
        cand_indexes = []
        cand_tokens = []
        for (i, token) in enumerate(input_tokens):
            if token in self.special_tokens:
                continue

            clean_token = self._clean_token(token)

            if len(cand_indexes) >= 1 and self._is_sub_word(token):
                cand_indexes[-1].append(i)
                # add the token without the initial `##`
                cand_tokens[-1] = cand_tokens[-1] + clean_token
            else:
                cand_indexes.append([i])
                cand_tokens.append(clean_token)

        assert len(cand_indexes) == len(cand_tokens)

        mask_indecies = []
        for i in range(len(cand_indexes)):
            for mword_list in self.mask_word_list:
                if i+len(mword_list)-1 >= len(cand_indexes):
                    continue
                # all the words in the masked word list candidate are in the sequence.
                if all([cand_tokens[i+j] == mword_list[j] for j in range(len(mword_list))]):
                    mask_indecies += [k for j in range(len(mword_list)) for k in cand_indexes[i+j]]

        if len(mask_indecies) == 0:
            # logger.info(f'Did not find any matches')
            return super()._whole_word_mask(input_tokens, max_predictions)

        mask_labels = [1 if i in mask_indecies else 0 for i in range(len(input_tokens))]
        # logger.info(f'mask_labels: {mask_labels}')

        return mask_labels

