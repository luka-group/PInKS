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
import IPython
import hydra
import omegaconf

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
    for idx,(word,tag) in enumerate(result):
        tmp_result = [list(ele) for ele in result]
        if (tag in NOUN_CODES) or (tag in ADJECTIVE_CODES):
            tmp_result[idx][0]="[MASK]"
            new_sent_masked=' '.join(word[0] for word in tmp_result)
            
            unmasked_list=list(unmasker(new_sent_masked))[:3]
            
            aug_masked_sents[new_sent_masked]=unmasked_list
    return aug_masked_sents
    
        

# sent="Glass is used for drinking water unless it is broken"
# temp_dict=fill_mask(sent)


# def return_augmented_dict(df):
    
#     return aug_sents
        



@hydra.main(config_path="../Configs", config_name="data_aug_config")
def main(config: omegaconf.dictconfig.DictConfig):
    try:
        with open(config.augmented_dataset_path) as f:
            aug_sents = json.load(f)
    except:
        print("Searched file at="+str(config.augmented_dataset_path))
        aug_sents={}
     
    df=pd.read_csv(config.filtered_dataset_path) 
    count=0
    for index, row in tqdm(df.iterrows()):
        if row['text'] not in aug_sents:
            try:
                aug_sents[row['text']]=fill_mask(row['text'])
                count+=1
            except Exception as e:
                print(e)
                continue
        if index%20000==0:
            with open(config.augmented_dataset_path, "w") as outfile:
                json.dump(aug_dataset_dict, outfile)
            print("Saved at index="+str(index))
            
    #Saving for the final time            
    with open(config.augmented_dataset_path, "w") as outfile:
        json.dump(aug_dataset_dict, outfile)
    
    print("Sentence augmented="+str(count))


if __name__ == '__main__':
    main()    



