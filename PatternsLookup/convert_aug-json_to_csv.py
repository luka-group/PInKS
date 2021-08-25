# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 20:07:01 2021

@author: Dell
"""


import logging
logger = logging.getLogger(__name__)
import IPython
import pandas as pd 
from tqdm import tqdm

import json  



with open(r"/nas/home/pkhanna/CQplus/Outputs/data_augmentation_lm/augmented_dataset.json") as f:
    aug_sents = json.load(f)


aug_sents_list=[]

for key, value in tqdm(aug_sents.items()):
    if aug_sents[key]:
        for key_masked, value_unmasked_sents in aug_sents[key].items():
            for unmasked_pred in value_unmasked_sents:
                aug_sents_list.append(unmasked_pred["sequence"])
                
        
            
aug_df = pd.DataFrame(aug_sents_list, columns =['Augmented Sentences'])

aug_df.to_csv('/nas/home/pkhanna/CQplus/Outputs/data_augmentation_lm/augmented_sents.csv')
    
        
    

