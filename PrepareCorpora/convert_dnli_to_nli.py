import os
import pathlib

import IPython
import pandas as pd
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
from functools import partial
from typing import Dict
import re

for name in tqdm(['train', 'test', 'dev']):
    df_dnli = pd.read_json(
        f"/nas/home/qasemi/CQplus/Outputs/Corpora/defeasible-nli/data/defeasible-nli/defeasible-snli/{name}.jsonl",
        lines=True
    ).apply(axis=1, func=lambda s: pd.Series({
        "context": s['Update'],
        "question": s['Hypothesis'] + s['Premise'],
        "label": 0 if s['UpdateType'] == 'weakener' else 1,
    })).to_csv(f'/nas/home/qasemi/CQplus/Outputs/Other_NLI/dnli_nli_{name}.csv', index=False)


