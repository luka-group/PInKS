import logging
import traceback
from itertools import chain
from typing import Callable, Union, List, Generator, Iterable, Dict, Any

import IPython
import omegaconf
import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl

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


def flatten_config(config: omegaconf.dictconfig.DictConfig) -> Dict[str, Union[str, int, float]]:
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


def unflatten_config(hparams: Dict[str, Union[str, int, float]]) -> Dict[str, Any]:
    update_dict = {}
    for k_bundle, v in hparams.items():
        elem = update_dict
        k_list = str(k_bundle).split('.')
        if len(k_list) > 1:
            for k in k_list[:-1]:
                if k not in elem:
                    elem[k] = {}
                elem = elem[k]
        elem[k_list[-1]] = v
    return update_dict


class PLModelDataTest:
    def __init__(self, model, data):
        self.model: pl.LightningDataModule = model
        self.data: pl.LightningModule = data

    def run(self):
        method_list = [method for method in dir(PLModelDataTest) if method.startswith('test_') is True]
        # self.test_train_dataloader()
        for method in method_list:
            logging.warning(f'Running {method}')
            getattr(PLModelDataTest, method)(self)

    def not_test_train_dataloader(self):
        self.data.setup('test')
        loader = iter(self.data.train_dataloader())
        i = 0
        tbar = tqdm(desc='training batch')
        while True:
            try:
                batch = next(loader)
                i += 1
                tbar.update()
            except StopIteration:
                return
            except Exception as e:
                logging.error('Something happened during loading training batch')
                IPython.embed()
                raise e

    def not_test_test_dataloader(self):
        loader = iter(self.data.test_dataloader())
        batch = ''
        i = 0
        tbar = tqdm(desc='testing batch')
        while True:
            try:
                batch = next(loader)
                i += 1
                tbar.update()
            except StopIteration:
                return
            except Exception as e:
                logging.error('Something happened during loading training batch')
                IPython.embed()
                raise e

    def not_test_one_epoch(self):
        self.data.setup('test')

    def test_epoch_end(self):
        self.data.setup('test')
        loader = iter(self.data.test_dataloader())
        outputs = []
        for i in tqdm(range(10)):
            try:
                batch = next(loader)
            except Exception as e:
                logging.error('Something happened during loading')
                IPython.embed()
                raise e
            try:
                outputs.append(
                    self.model.test_step(batch, i)
                )
            except Exception as e:
                logging.error('Something happened during training_step')
                IPython.embed()
                raise e

        try:
            self.model.extra_tag = "unittest"
            out = self.model.test_epoch_end(outputs)
        except Exception as e:
            logging.error('test_epoch_end Failed')
            IPython.embed()
            raise e

        # trainer = pl.Trainer(fast_dev_run=3600, gpus="0")
        # trainer.fit(model, datamodule=data)
        IPython.embed()
        exit()



