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
from nltk.corpus import wordnet as wn

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


@hydra.main(config_path="../Configs", config_name="filter_dataset_config")
def main(config: omegaconf.dictconfig.DictConfig):
    merged_df=pd.read_csv(config.merged_dataset)
    filtered_dataset=pd.DataFrame(columns=merged_df.columns)
    for index,row in merged_df.iterrows():
        if isQuestion(row['text']) or hasVerb(row['precondition']):
            filtered_dataset.append(row)   
    filtered_dataset.to_csv(config.output_path)

if __name__ == '__main__':
    main()