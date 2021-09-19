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

    


@hydra.main(config_path="../Configs", config_name="snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    """With SnorkelUtil"""
    omcs_df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    omcs_df['text'] = omcs_df['text'].astype(str)

    L_omcs, LFA_df=SnorkelUtil(omcs_df)
    examples_df=SnorkelUtil.returnExamples(L_omcs, LFA_df, omcs_df)
    examples_df.to_csv(config.output_examples)
    
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_omcs, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    omcs_df["label"] = label_model.predict(L=L_omcs, tie_break_policy="abstain")
    
    omcs_df=SnorkelUtil.addActionPrecondition(L_omcs, LFA_df, omcs_df)
    omcs_df = omcs_df[omcs_df.label != SnorkelUtil.ABSTAIN]
    
    omcs_df.to_csv(config.output_name)
    
    count = omcs_df["label"].value_counts()
    print("Label  Count")
    print(count)
    
    with open('LabelingMatrix.npy', 'wb') as f:
        np.save(f, L_omcs)
    
    


if __name__ == '__main__':
    main()

