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
from snorkel.labeling import LabelingFunction

from snorkel.labeling.model import LabelModel
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis

import nltk
from nltk.corpus import wordnet as wn

nltk.download("wordnet")


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

ENABLING_PATTERNS = [
    "{action} only if {precondition}.",
    "{precondition} (?:so|hence|consequently) {action}.",
    "{precondition} makes {action} possible.",
]

DISABLING_WORDS = [
    "unless",
]


ABSTAIN = -1
DISABLING = 0
ENABLING = 1


def pattern_exists(pattern,line):
    pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
    replacements = {k: REPLACEMENT_REGEX[k] for k in pattern_keys}    
    regex_pattern = pattern.format(**replacements)
    m_list = re.findall(regex_pattern, line)
    
    
    for m in m_list:
        match_full_sent = line
        for sent in line:
            if all([ps in sent for ps in m]):
                match_full_sent = sent
    
        match_dict = dict(zip(pattern_keys, m))
        if 'negative_precondition' in pattern_keys:
                    if not(any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS])):
                        return False
    if len(m_list)>0:
        return True
    return False


def keyword_lookup(x, keyword, label):
    pat="{action} " +  keyword + " {precondition}."
    if pattern_exists(pat,x.text):
        return label
    else:
        return ABSTAIN


def make_keyword_lf(keyword, label):
    lf_name=keyword.replace(" ","_") + f"_{label}"
    return LabelingFunction(
        name=lf_name,
        f=keyword_lookup,
        resources=dict(keyword=keyword, label=label),
    )


lfs=[]

pos_conj = {'only if', 'contingent upon', 'given', 'if',"in case", "in the case that", "in the event", "on condition", "on the assumption",
            "on these terms", "subject to", "supposing", "with the proviso"}

neg_conj = {"but", "except", "except for", "excepting that", "if not", "lest", "saving", "without"}


for p_conj in pos_conj:
    lfs.append(make_keyword_lf(p_conj,ENABLING))
    
    
for n_conj in neg_conj:
   lfs.append(make_keyword_lf(n_conj,DISABLING))   
    


@labeling_function()
def disabling1(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
        if pattern_exists(pat,x.text):
            return DISABLING
    return ABSTAIN


              
@labeling_function()
def enabling_makespossible(x):
    pat="{precondition} makes {action} possible."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN


lfs.extend([disabling1,enabling_makespossible])

# lfs = [disabling1, disabling2, enabling_onlyif, enabling_so_hence_conseq, enabling_makespossible]


@hydra.main(config_path="../Configs", config_name="snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    omcs_df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    omcs_df['text'] = omcs_df['text'].astype(str)
    
    applier = PandasLFApplier(lfs)
    L_omcs = applier.apply(omcs_df)
    
    print(LFAnalysis(L_omcs, lfs).lf_summary())
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_omcs, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    omcs_df["label"] = label_model.predict(L=L_omcs, tie_break_policy="abstain")
    
    omcs_df = omcs_df[omcs_df.label != ABSTAIN]
    
    print(config.output_name)
    omcs_df.to_csv(config.output_name)
    
    count = omcs_df["label"].value_counts()
    print("Label  Count")
    print(count)
    


if __name__ == '__main__':
    main()





