# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 23:17:50 2021

@author: Dell
"""
import os
import pathlib
from typing import Dict, Generator

import pandas as pd
import IPython
import hydra
import omegaconf
import json
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



# @labeling_function()
# def is_a_kind_of(x):
#     return NOT_RELEVANT if "is a kind of" in x.text.lower() else ABSTAIN

@labeling_function()
def disabling1(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
        if pattern_exists(pat,x.text):
            return DISABLING
    return ABSTAIN


@labeling_function()
def disabling2(x):
    pat=r"{negative_precondition} (?:so|hence|consequently) {action}\."
    if pattern_exists(pat,x.text):
        return DISABLING
    else:
        return ABSTAIN
    
@labeling_function()
def disabling3(x):
    pat=r'{precondition} makes {action} impossible.'
    if pattern_exists(pat,x.text):
        return DISABLING
    else:
        return ABSTAIN
        
        
@labeling_function()
def enabling_onlyif(x):
    pat="{action} only if {precondition}."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
        
@labeling_function()
def enabling_so_hence_conseq(x):
    pat="{precondition} (?:so|hence|consequently) {action}."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
              
@labeling_function()
def enabling_makespossible(x):
    pat="{precondition} makes {action} possible."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN


lfs = [disabling1, disabling2, disabling3, enabling_onlyif, enabling_so_hence_conseq, enabling_makespossible]


def extract_all_sentences_df(config: omegaconf.dictconfig.DictConfig):    
    # assert config.predicate == '*', f'{config.predicate}'
    logger.info(f'loading json from {config.ascent_path}')
    output = []
    all_sents=[]
    
    pbar_concept = tqdm(desc='concepts')
    # pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True, chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()

            # create sources lut
            # sent_dict = {}
            for k, s in concept['sentences'].items():
                all_sents.append(s['text'].replace('\n', ' '))
                
                


    logger.info(f'converting to pandas')
    print("Output Path:")
    print(config.output_names.extract_all_sentences_df)
    
    df = pd.DataFrame(all_sents, columns =['text'])
    df.to_csv(config.output_names.extract_all_sentences_df)
    # df.to_json('all_sentences.json')
    # df.to_json(config.output_names.extract_all_sentences, orient='records', lines=True)
    # return df


def process_all_sentences_snorkel(config: omegaconf.dictconfig.DictConfig):
    all_sents_path = pathlib.Path(os.getcwd())/pathlib.Path(config.output_names.extract_all_sentences_df)

    assert all_sents_path.exists(), all_sents_path
    
    df = pd.read_csv(all_sents_path)
    df['text'] = df['text'].astype(str)
    
    applier = PandasLFApplier(lfs)
    L_data = applier.apply(df)
    
    print(LFAnalysis(L_data, lfs).lf_summary())
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_data, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    df["label"] = label_model.predict(L=L_data, tie_break_policy="abstain")
    
    df = df[df.label != ABSTAIN]
    
    print(config.output_names.process_all_sentences_df)
    df.to_csv(config.output_names.process_all_sentences_df)
    
    count = df["label"].value_counts()
    print("Label  Count")
    print(count)



@hydra.main(config_path="../Configs", config_name="process_ascent_snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    logger.warning(f'Config: {config}')

    if config.method == 'extract_all_sentences_df':
        extract_all_sentences_df(config)
    elif config.method == 'process_all_sentences_snorkel':
        process_all_sentences_snorkel(config)
    # pd.read_json(config)
    # extract_usedfor_assertions(config)
    # extract_usedfor_sentences(config)


if __name__ == '__main__':
    main()
