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


question_start_words = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]

VERB_CODES = {
    'VB',  # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
}


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


def containsIf(text):
    if_not_pat="{action} if not {precondition}"
    if pattern_exists(if_not_pat,text):
        return False
    elif pattern_exists("{action} if {precondition}",text):
        return True
    return False
    

def containsBut(text):
    pat="{action} but {precondition}"
    if pattern_exists(pat,text):
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
        if not(isQuestion(row['text'])) and hasVerb(row['precondition']) and isEnglish(row['text']) and not(containsIf(row['text'])) and not(containsBut(row['text'])):
            new_row = {"text": row['text'], "action": row['action'], "precondition": row['precondition'], "label":row['label']}
            filtered_dataset = filtered_dataset.append(new_row, ignore_index = True)
            count+=1
    # print("Filtered True count="+str(count))
    print("Filtered len="+str(len(filtered_dataset)))
    filtered_dataset.to_csv(config.output_path)
    
    count = filtered_dataset["label"].value_counts()
    print("Label  Count")
    print(count)
    

if __name__ == '__main__':
    main()
    
    
    
    

    
