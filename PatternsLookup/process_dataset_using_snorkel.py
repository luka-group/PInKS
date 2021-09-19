# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:30:05 2021

@author: Dell
"""
import pandas as pd 

import os

import IPython
import hydra
import omegaconf
import json
import re
from tqdm import tqdm
import numpy as np

from Patterns import PatternUtils

from snorkel.labeling import labeling_function
from snorkel.labeling import LabelingFunction

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")



import logging
logger = logging.getLogger(__name__)


from SnorkelUtil import SnorkelUtil

    


@hydra.main(config_path="../Configs", config_name="process_dataset_using_snorkel")
def main(config: omegaconf.dictconfig.DictConfig):
    """With SnorkelUtil"""
    df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    df['text'] = df['text'].astype(str)

    L, LFA_df=SnorkelUtil(df)
    examples_df=SnorkelUtil.returnExamples(L, LFA_df, df)
    examples_df.to_csv(config.output_examples)
    
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    df["label"] = label_model.predict(L=L, tie_break_policy="abstain")
    
    df=SnorkelUtil.addActionPrecondition(L, LFA_df, df)
    df = df[df.label != SnorkelUtil.ABSTAIN]
    
    df.to_csv(config.output_name)
    
    count = df["label"].value_counts()
    print("Label  Count")
    print(count)
    
    with open('LabelingMatrix.npy', 'wb') as f:
        np.save(f, L)
    
    


if __name__ == '__main__':
    main()

