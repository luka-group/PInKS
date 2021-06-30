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

from Patterns import PatternUtils

from snorkel.labeling import labeling_function

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier



import logging
logger = logging.getLogger(__name__)



FACT_REGEX = r'([a-zA-Z0-9_\-\\\/\+\* \'’%]{10,})'

REPLACEMENT_REGEX = {
        'action': FACT_REGEX,
        'precondition': FACT_REGEX,
        'negative_precondition': FACT_REGEX,
        'precondition_action': FACT_REGEX,
        'any_word': r'[^ \[]{,10}',
        'ENB_CONJ': r'(?:so|hence|consequently|thus|therefore|'
                    r'as a result|thus|accordingly|because of that|'
                    r'as a consequence|as a result)',
    }

# pattern = "{action} unless {precondition}"

NEGATIVE_WORDS = [
    ' not ',
    ' cannot ',
    'n\'t ',
    ' don\\u2019t ',
    ' doesn\\u2019t ',
]



SINGLE_SENTENCE_DISABLING_PATTERNS1 = [
    r"^{action} unless {precondition}\.",
    r"\. {action} unless {precondition}\.",
    r"^{any_word} unless {precondition}, {action}\.",
    r"^{any_word} unless {precondition}, {action}\.",
]

SINGLE_SENTENCE_DISABLING_PATTERNS2 = [
    r"{negative_precondition} (?:so|hence|consequently) {action}\.",
]




ABSTAIN = -1
NOT_RELEVANT = 0
RELEVANT = 1


def pattern_exists(pattern,sent):
    pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
    replacements = {k: REPLACEMENT_REGEX[k] for k in pattern_keys}    
    regex_pattern = pattern.format(**replacements)
    m_list = re.findall(regex_pattern, sent)
    
    if 'negative_precondition' in pattern_keys:
                if not(any([nw in sent for nw in PatternUtils.NEGATIVE_WORDS])):
                    return False
    if len(m_list)>0:
        return True
    return False



@labeling_function()
def is_a_kind_of(x):
    return NOT_RELEVANT if "is a kind of" in x.text.lower() else ABSTAIN

@labeling_function()
def single_sent_disabling_pat1(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
        if pattern_exists(pat,x.text):
            return RELEVANT
        else:
            return ABSTAIN


@labeling_function()
def single_sent_disabling_pat2(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS2:
        if pattern_exists(pat,x.text):
            return RELEVANT
        else:
            return ABSTAIN
              

lfs = [single_sent_disabling_pat1, single_sent_disabling_pat2,is_a_kind_of]


@hydra.main(config_path="../Configs", config_name="snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    omcs_df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    
    applier = PandasLFApplier(lfs)
    L_omcs = applier.apply(omcs_df)
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_omcs, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    omcs_df["label"] = label_model.predict(L=L_omcs, tie_break_policy="abstain")
    
    omcs_df = omcs_df[omcs_df.label != ABSTAIN]
    
    omcs_df.to_csv(config.output_name)
    
    count = omcs_df["label"].value_counts()
    print("Label  Count")
    print(count)
    


if __name__ == '__main__':
    main()





