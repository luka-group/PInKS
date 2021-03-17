from itertools import chain
from typing import Callable, Union, List, Generator, Iterable

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
