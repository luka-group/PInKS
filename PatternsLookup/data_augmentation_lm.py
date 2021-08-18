# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:48:13 2021

@author: Dell
"""

import os
import pathlib
import re
from typing import Dict, Generator
import json
from tqdm import tqdm
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
from nltk.corpus import wordnet as wn
from transformers import pipeline

nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')


unmasker = pipeline('fill-mask', model='bert-base-uncased')


NOUN_CODES = {'NN','NNS','NNPS','NNP'}
ADJECTIVE_CODES = {"JJ","JJR", "JJS"}

def fill_mask(text):
    aug_masked_sents={}
    text = nltk.word_tokenize(text)
    result = nltk.pos_tag(text)
    print(result)
    for idx,(word,tag) in enumerate(result):
        tmp_result = [list(ele) for ele in result]
        if (tag in NOUN_CODES) or (tag in ADJECTIVE_CODES):
            tmp_result[idx][0]="[MASK]"
            new_sent_masked=' '.join(word[0] for word in tmp_result)
            print("Masked sent="+new_sent_masked)
            unmasked_list=list(unmasker(new_sent_masked))[:3]
            print(unmasked_list)
            aug_masked_sents[new_sent_masked]=unmasked_list
    return aug_masked_sents
    
        

# sent="Glass is used for drinking water unless it is broken"
# temp_dict=fill_mask(sent)


def return_augmented_dict(df):
    aug_sents={}
    for index, row in tqdm(df.iterrows()):
        aug_sents[row['text']]=fill_mask(row['text'])
    return aug_sents
        


df=pd.read_csv("/nas/home/pkhanna/CQplus/Outputs/filter_dataset/filtered_dataset.csv") 
print(len(df))       
aug_dataset_dict=return_augmented_dict(df)


with open("/nas/home/pkhanna/CQplus/Outputs/augmented_dataset.json", "w") as outfile:
    json.dump(aug_dataset_dict, outfile)

