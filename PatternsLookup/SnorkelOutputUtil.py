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

from SnorkelUtil import SnorkelUtil


import logging
logger = logging.getLogger(__name__)



class ProcessOutputUtil():

    @staticmethod
    def filter_dataset(merged_df):
        question_start_words = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]
        VERB_CODES = {
            'VB',  # Verb, base form
            'VBD',  # Verb, past tense
            'VBG',  # Verb, gerund or present participle
            'VBN',  # Verb, past participle
            'VBP',  # Verb, non-3rd person singular present
            'VBZ',  # Verb, 3rd person singular present
        }

        column_names = ["text", "action", "precondition","label"]
        filtered_dataset=pd.DataFrame(columns=column_names)

        count=0
        for index,row in tqdm(merged_df.iterrows()):
            if not(ProcessOutputUtil.isQuestion(row['text'])) and ProcessOutputUtil.hasVerb(row['precondition']) and ProcessOutputUtil.isEnglish(row['text']):
                new_row = {"text": row['text'], "action": row['action'], "precondition": row['precondition'], "label":row['label']}
                filtered_dataset = filtered_dataset.append(new_row, ignore_index = True)
                count+=1
        # print("Filtered True count="+str(count))
        print("Filtered len="+str(len(filtered_dataset)))
        filtered_dataset.to_csv(config.output_path)
        
        count = filtered_dataset["label"].value_counts()
        print("Label  Count")
        print(count)
    
    @staticmethod
    def isQuestion(text):
    text=text.strip()
    if ('?' in text) or (text.split()[0].lower() in question_start_words):
        return True
    return False

    @staticmethod
    def hasVerb(text):
        text = nltk.word_tokenize(text)
        result = nltk.pos_tag(text)
        for tags in result:
            if tags[1] in VERB_CODES:
                return True
        return False

    @staticmethod
    def isEnglish(text):
        if detect(text)=='en':
            return True
        return False

    @staticmethod
    def containsIf(text):
        if_not_pat="{action} if not {precondition}"
        if pattern_exists(if_not_pat,text):
            return False
        elif pattern_exists("{action} if {precondition}",text):
            return True
        return False
        
    @staticmethod
    def containsBut(text):
        pat="{action} but {precondition}"
        if pattern_exists(pat,text):
            return True
        return False


