from itertools import chain
from typing import Callable, Union, List, Generator, Iterable, Dict

import IPython
import omegaconf
import torch
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm


class ClassificationDataset(Dataset):
    def __init__(self, instances):
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        # assert idx < self.__len__(), f'{idx} >= {self.__len__()}'
        return self.instances[idx]


def my_list_flatmap(f, items):
    return chain.from_iterable(map(f, items))


def my_df_flatmap(df: pd.DataFrame,
                  func: Callable[
                      [pd.Series],
                      Union[Iterable[pd.Series], Generator[pd.Series, None, None]]]
                  ) -> pd.DataFrame:
    rows = []
    for index, row in tqdm(df.iterrows()):
        multrows = func(row)
        for rr in multrows:
            rows.append(rr)
    return pd.DataFrame.from_records(rows)


def config_to_hparams(config) -> Dict[str, Union[str, int, float]]:
    hparams = {}

    def _setup_hparams(d_conf, prefix: str = ''):
        key = lambda k: (f'{prefix}.' if prefix != '' else '') + f'{k}'
        for k, v in dict(d_conf).items():
            if any([isinstance(v, t) for t in [int, float, str, bool, torch.Tensor]]):
                hparams[key(k)] = v
            elif isinstance(v, omegaconf.listconfig.ListConfig):
                hparams[key(k)] = v
            elif v is None:
                hparams[key(k)] = ''
            elif isinstance(v, omegaconf.dictconfig.DictConfig):
                _setup_hparams(v, prefix=k)
            else:
                raise ValueError(f'invalid config type: [{k}]={v} with type {type(v)}')

    _setup_hparams(config)
    return hparams