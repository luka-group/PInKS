# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 19:25:11 2021

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
import re

import logging
logger = logging.getLogger(__name__)


df=pd.read_csv('/nas/home/pkhanna/CQplus/Outputs/merge_matches/final_merged.csv')


for row,index in tqdm(df.iterrows()):
    if row['action']==-1 or row['precondition']==-1:
        print("Action/precondition=-1")
        print(row['text'])
        print(row['label'])
        
    if row['label']!=0 and row['label']!=1:
        print("Wrong label!")
        print(row['text'])
        print(row['action'])
        print(row['precondition'])
        print(row['label'])
        