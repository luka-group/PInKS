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

import nltk
nltk.download('punkt')
from nltk.corpus import wordnet as wn

from langdetect import detect

nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')


question_start_words = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]

VERB_CODES = {
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
}

def isQuestion(text):
    text=text.strip()
    if text[-1]=='?' or text.split()[0].lower() in question_start_words:
        return True
    return False

def hasVerb(text):
    text = nltk.word_tokenize(text)
    result = nltk.pos_tag(text)
    for tags in result:
        if tags[1] in VERB_CODES:
            return True
    return False


def isEnglish(text):
    if detect(text)=='en':
        return True
    return False


@hydra.main(config_path="../Configs", config_name="filter_dataset_config")
def main(config: omegaconf.dictconfig.DictConfig):
    merged_df=pd.read_csv(config.merged_dataset)
    print("Merged DF len="+str(len(merged_df)))
    column_names = ["text", "action", "precondition","label"]
    filtered_dataset=pd.DataFrame(columns=column_names)
    count=0
    for index,row in tqdm(merged_df.iterrows()):
        if not(isQuestion(row['text'])) and hasVerb(row['precondition']) and isEnglish(row['text']):
            new_row = {"text": row['text'], "action": row['action'], "precondition": row['precondition'], "label":row['label']}
            filtered_dataset = filtered_dataset.append(new_row, ignore_index = True)
            count+=1
    # print("Filtered True count="+str(count))
    print("Filtered len="+str(len(filtered_dataset)))
    filtered_dataset.to_csv(config.output_path)

if __name__ == '__main__':
    main()
    
    
    
    

    
