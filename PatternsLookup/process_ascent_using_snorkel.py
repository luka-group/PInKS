# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 23:17:50 2021

@author: Dell
"""
import os
import pathlib
import re
from typing import Dict, Generator

import pandas as pd
import IPython
import hydra
import omegaconf
import json
from tqdm import tqdm
from Patterns import PatternUtils
import numpy as np

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
AMBIGUOUS=2


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


@labeling_function()
def ambiguous_pat(x):
    pat="{precondition} (?:so|hence|consequently) {action}."
    if pattern_exists(pat,x.text):
        return AMBIGUOUS
    else:
        return ABSTAIN


lfs.extend([disabling1, enabling_makespossible, ambiguous_pat])



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


#Return (precondition, action) pair.
def get_precondition_action(pattern,line):
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
        return match_dict['precondition'], match_dict['action']


#Adds Action, Precondtion columns to df.
def addActionPrecondition(L, LFA_df, df):
    actions=[]
    preconditions=[]
    lfs_names=list(LFA_df.index)
    for index,row in df.iterrows():
        position = np.argmax(L[index,:] > -1)
        action=-1
        precondition=-1
        if position==0 and L[index,position]==-1:
            action=-1
            precondition=-1
        else:
            conj=lfs_names[position][:-2].replace("_"," ")
            # print(conj)
            if conj=="ambiguous pat":
                conj="(?:so|hence|consequently)"
            pat="{action} " +  conj + " {precondition}."
            if conj=="makespossible":
                pat="{precondition} makes {action} possible."
                
            try:
                precondition, action= get_precondition_action(pat,row['text'])
            except Exception as e:
                print(e)
                print("pattern="+pat)
                print("text="+row['text'])
        actions.append(action)
        preconditions.append(precondition)
    print("DF len="+str(len(df)))
    print("Actions len="+str(len(actions)))
    df['Action']=actions
    df['Precondition']=preconditions
    return df



def process_all_sentences_snorkel(config: omegaconf.dictconfig.DictConfig):
    all_sents_path = pathlib.Path(os.getcwd())/pathlib.Path(config.output_names.extract_all_sentences_df)

    assert all_sents_path.exists(), all_sents_path
    
    print("Processing via Snorkel")
    
    df = pd.read_csv(all_sents_path)
    df['text'] = df['text'].astype(str)
    
    applier = PandasLFApplier(lfs)
    L_data = applier.apply(df)
    
    LFA_df=LFAnalysis(L_data, lfs).lf_summary().copy()
    
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_data, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    df["label"] = label_model.predict(L=L_data, tie_break_policy="abstain")
    
    df=addActionPrecondition(L_data, LFA_df, df)
    
    df = df[df.label != ABSTAIN]
    

    
    count = df["label"].value_counts()
    print("Label  Count")
    print(count)
    
    print(config.output_names.process_all_sentences_snorkel)
    df.to_csv(config.output_names.process_all_sentences_snorkel)
    
    print("LF_Analysis")
    print(LFAnalysis(L_data, lfs).lf_summary())
    




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
