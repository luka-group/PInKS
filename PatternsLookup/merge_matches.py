# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 20:27:11 2021

@author: Dell
"""

import os

import IPython
import hydra
import omegaconf
import json

from tqdm import tqdm

from Patterns import PatternUtils
import pandas as pd

import logging
logger = logging.getLogger(__name__)


"""To Do: Add code to disambiguate the the AMBIGUOUS pattern label and make neg_precondition as positive.

If neg word exists in precondition: change precondition to a positive sentence and label it as disabling.
else: label it as enabling.

"""


ABSTAIN = -1
DISABLING = 0
ENABLING = 1
AMBIGUOUS=2



def disambiguate(line):    
    pattern = "{precondition} (?:so|hence|consequently) {action}."
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

        if any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS]):
            match_full_sent = PatternUtils.make_sentence_positive(match_full_sent)
            match_dict['precondition'] = PatternUtils.make_sentence_positive(match_dict['negative_precondition'])
            return match_dict['precondition'], DISABLING

        return match_dict['precondition'], ENABLING


def process_df(df,text,actions,preconditions,labels):
    for index,row in df.iterrows():
        action=row["Action"]
        precondition=row["Precondition"]
        label=row["label"]
        if label==2:
            precondition, label = disambiguate(row["text"])
        text.append(row["text"])
        actions.append(action)
        preconditions.append(precondition)
        labels.append(label)
    return
    


@hydra.main(config_path="../Configs", config_name="merge_matches_config")
def main(config: omegaconf.dictconfig.DictConfig):
    
    path1=config.matches_path1
    path2=config.matches_path2
    
    text=[]
    actions=[]
    preconditions=[]
    labels=[]
    
    df1=pd.read_csv(config.matches_path1)
    df2=pd.read_csv(config.matches_path2)
    
    process_df(df1,text,actions,preconditions,labels)
    process_df(df2,text,actions,preconditions,labels)
    
    final_df = pd.DataFrame(list(zip(text,actions,preconditions,labels)), columns =['text', 'action','precondition','label'])
    
    final_df.to_csv(config.output_path)

    

if __name__ == '__main__':
    main()