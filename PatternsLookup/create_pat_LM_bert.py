# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 19:58:07 2021

@author: Dell
"""


from transformers import pipeline


import logging
logger = logging.getLogger(__name__)
import IPython
import hydra
import omegaconf
import pandas as pd 
from tqdm import tqdm

import json  


unmasker = pipeline('fill-mask', model='bert-base-uncased')

all_conj={'only if', 'contingent upon', 'if',"in case", "in the case that", "in the event", "on condition", "on the assumption",
            "on these terms",  "supposing", "with the proviso",
            "except", "except for", "excepting that", "if not", "lest",  "without"}

possible_replacements={}


for conj in all_conj:
    possible_replacements[conj]=set()

def addMaskedConj(sent):
    for conj in all_conj:
        if conj in sent:
            sent=sent.replace(conj,"[MASK]",1)
            unmasked_words=list(unmasker(sent))
            for new_word in unmasked_words[:3]:
                if new_word['token_str']!=conj:
                    possible_replacements[conj].add(new_word['token_str'])
            


@hydra.main(config_path="../Configs", config_name="create_pat_LM_config")
def main(config: omegaconf.dictconfig.DictConfig):
    df=pd.read_csv(config.corpus_path)
    for index,row in tqdm(df.iterrows()):
        addMaskedConj(row['text'])
    print(config.output_name)
    with open(config.output_name, "w") as outfile: 
        json.dump(possible_replacements, outfile)
        
        
        
if __name__ == '__main__':
    main()    