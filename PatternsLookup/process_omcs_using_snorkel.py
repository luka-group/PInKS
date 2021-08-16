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




FACT_REGEX = r'([a-zA-Z0-9_\-\\\/\+\* \'"’%]{10,})'
EVENT_REGEX = r'([a-zA-Z0-9_\-\\\/\+\*\. \'’%]{10,})'

REPLACEMENT_REGEX = {
        'action': FACT_REGEX,
        'event': EVENT_REGEX,
        'negative_action': FACT_REGEX,
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
    
    # print(pattern_keys)
    # print(replacements)
    # print(regex_pattern)
    # print(m_list)
    
    for m in m_list:
        match_full_sent = line
        for sent in line:
            if all([ps in sent for ps in m]):
                match_full_sent = sent
    
        match_dict = dict(zip(pattern_keys, m))
        if 'negative_precondition' in pattern_keys:
                    if any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS]):
                        return True
                    else:
                        return False

        # if 'negative_action' in pattern_keys:
        #     if any([nw in match_dict['negative_action'] for nw in PatternUtils.NEGATIVE_WORDS]):
        #         return True
        #     else:
        #         return False
    if len(m_list)>0:
        return True
    return False


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
        
        action=""
        precondition=""
        
        
        if 'negative_precondition' in pattern_keys:
            precondition=match_dict['negative_precondition']
        else:
            precondition=match_dict['precondition']
            
        if 'event' in pattern_keys:
            action=match_dict['event']
        else:
            action=match_dict['action']

        # if 'negative_action' in pattern_keys:
        #     action=match_dict['negative_action']
        # else:
        #     action=match_dict['action']
            
        
        return precondition, action
    




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

pos_conj = {'only if', 'contingent upon', 'if',"in case", "in the case that", "in the event", "on condition", "on the assumption",
            "on these terms",  "supposing", "with the proviso"}

neg_conj = {"except", "except for", "excepting that", "if not", "lest"}

@labeling_function()
def unless_0(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
        if pattern_exists(pat,x.text):
            return DISABLING
    return ABSTAIN


@labeling_function()
def but_0(x):
    pat="{action} but {negative_precondition}"
    if pattern_exists(pat,x.text):
        return DISABLING
    else:
        return ABSTAIN

              
@labeling_function()
def makes_possible_1(x):
    pat="{precondition} makes {action} possible."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
    
@labeling_function()
def to_understand_event_1(x):
    pat = r'To understand the event "{event}", it is important to know that {precondition}.'
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
  
    
@labeling_function()
def statement_is_true_1(x):
    pat = r'The statement "{event}" is true because {precondition}.'
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
    
    
# @labeling_function()
# def ambiguous_pat_2(x):
#     pat="{precondition} (?:so|hence|consequently) {action}."
#     if pattern_exists(pat,x.text):
#         return AMBIGUOUS
#     else:
#         return ABSTAIN


enabling_dict={}
disabling_dict={}

disabling_dict={
    'but' : "{action} but {negative_precondition}",
    'unless' : "{action} unless {precondition}",
    }


enabling_dict={
    'makes possible' : "{precondition} makes {action} possible.",
    'to understand event' : r'To understand the event "{event}", it is important to know that {precondition}.',
    'statement is true' : r'The statement "{event}" is true because {precondition}.',
    }




for p_conj in pos_conj:
    lfs.append(make_keyword_lf(p_conj,ENABLING))
    enabling_dict[p_conj]="{action} " +  p_conj + " {precondition}."

lfs.extend([makes_possible_1, to_understand_event_1, statement_is_true_1])
    
for n_conj in neg_conj:
   lfs.append(make_keyword_lf(n_conj,DISABLING)) 
   disabling_dict[n_conj]="{action} " +  n_conj + " {precondition}."
    

# lfs.extend([unless_0, but_0, ambiguous_pat_2])
lfs.extend([unless_0, but_0])



def returnExamples(L, LFA_df, omcs_df):
    lfs_names=list(LFA_df.index)
    df_data=None
    df=pd.DataFrame()
    N=100
    for index,row in LFA_df.iterrows():
        s_no=int(row['j'])
        label=int(index[-1])

        pat_matches=L[:, s_no] == label
        match_count=sum(bool(x) for x in pat_matches)
        tmp_list=list(omcs_df.iloc[L[:, s_no] == label].sample(min(match_count,N), random_state=1)['text'])
        if len(tmp_list)<N:
            tmp_list += [0] * (N - len(tmp_list))
        df[str(index)]=tmp_list
    return df




#Adds Action, Precondtion columns to df.
def addActionPrecondition(L, LFA_df, df):
    actions=[]
    preconditions=[]
    lfs_names=list(LFA_df.index)
    for index,row in df.iterrows():
        valid_positions=L[index,:] > -1
        # position = np.argmax(L[index,:] > -1)
        action=-1
        precondition=-1
        label=row["label"]
        
        if not(np.any(valid_positions)) or label==ABSTAIN:
            action=-1
            precondition=-1
        else:
            if label==ENABLING:
                position = np.argmax(L[index,:] == ENABLING)
                conj=lfs_names[position][:-2].replace("_"," ")
                pat=enabling_dict[conj]
            elif label==DISABLING:
                position = np.argmax(L[index,:] == DISABLING)
                conj=lfs_names[position][:-2].replace("_"," ")
                pat=disabling_dict[conj]
            else:                                                       #AMBIGUOUS PATTERN
                pat="{precondition} (?:so|hence|consequently) {action}."
            # position = np.argmax(L[index,:] > -1)
            # conj=lfs_names[position][:-2].replace("_"," ")
            # # print(conj)
            # if conj=="ambiguous pat":
            #     conj="(?:so|hence|consequently)"
            # pat="{action} " +  conj + " {precondition}."
            # if conj=="makespossible":
            #     pat="{precondition} makes {action} possible."
                
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
            
            
    



@hydra.main(config_path="../Configs", config_name="snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    omcs_df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    omcs_df['text'] = omcs_df['text'].astype(str)
    
    applier = PandasLFApplier(lfs)
    # global L_omcs
    L_omcs = applier.apply(omcs_df)
    
    # print(type(L_omcs))
    # print(L_omcs)
    
    
    # L_matrix=np.copy(L_omcs)
    
    print(LFAnalysis(L_omcs, lfs).lf_summary())
    print(type(LFAnalysis(L_omcs, lfs).lf_summary()))
    
    
    LFA_df=LFAnalysis(L_omcs, lfs).lf_summary().copy()
    
    examples_df=returnExamples(L_omcs, LFA_df, omcs_df)
    examples_df.to_csv(config.output_examples)
    
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_omcs, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    omcs_df["label"] = label_model.predict(L=L_omcs, tie_break_policy="abstain")
    
    
    omcs_df=addActionPrecondition(L_omcs, LFA_df, omcs_df)
    
    
    omcs_df = omcs_df[omcs_df.label != ABSTAIN]
    
    omcs_df.to_csv(config.output_name)
    
    count = omcs_df["label"].value_counts()
    print("Label  Count")
    print(count)
    
    
    with open('LabelingMatrix.npy', 'wb') as f:
        np.save(f, L_omcs)
    


if __name__ == '__main__':
    main()

