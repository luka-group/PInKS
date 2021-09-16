import functools
from functools import partial
from typing import Callable, Dict, List, Optional

import datasets
import pandas as pd
from transformers import AutoTokenizer

from Models.DataModules.ModMLMData import ModMLMDataModule, DataCollatorForPreconditionWordMask


class MaskedNLIDataModule(ModMLMDataModule):
    @staticmethod
    def _get_preprocess_func_4_model() -> Callable[[str, str], str]:

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
        self.tokenizer = AutoTokenizer.from_pretrained(
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
                    self.tokenizer,
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

        self.data_collator = DataCollatorForPreconditionWordMask(
            tokenizer=self.tokenizer,
            # mlm_probability=self.config.data_module.mlm_probability
        )

        self.train_dataset = all_tokenized['train'].remove_columns(columns_names['train'])
        self.test_dataset = all_tokenized['test'].remove_columns(columns_names['test'])


# class ModifiedDataCollatorForPreconditionWordMask(DataCollatorForPreconditionWordMask):
#     def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) \
#             -> Dict[str, torch.Tensor]:
#         # d = LMDataModule.__call__(self, examples)
#         # IPython.embed()
#         d = super(DataCollatorForPreconditionWordMask, self).__call__(examples)
#         # exit()
#         return d
