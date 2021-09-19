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



import logging
logger = logging.getLogger(__name__)






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


# ABSTAIN = -1
# DISABLING = 0
# ENABLING = 1
# AMBIGUOUS=2




class SnorkelUtil():
    
    ABSTAIN = -1
    DISABLING = 0
    ENABLING = 1
    # AMBIGUOUS=2
    
    
    def __init__(self,df):
        
        
        
        self.pos_conj = {'only if', 'contingent upon', 'if',"in case", "in the case that", "in the event", "on condition", "on the assumption",
                    "on these terms",  "supposing", "with the proviso"}
        self.neg_conj = {"except", "except for", "excepting that", "if not", "lest",  "without"}
        
        self.lfs=[]
            
        
        @labeling_function()
        def if_0(x):
            pat="{action} if not {precondition}"
            if SnorkelUtil.pattern_exists(pat,x.text):
                return SnorkelUtil.DISABLING
            elif SnorkelUtil.pattern_exists("{action} if {precondition}",x.text):
                return SnorkelUtil.ENABLING
            else:
                return SnorkelUtil.ABSTAIN
        
        @labeling_function()
        def unless_0(x):
            for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
                if SnorkelUtil.pattern_exists(pat,x.text):
                    return SnorkelUtil.DISABLING
            return SnorkelUtil.ABSTAIN
        
        
        @labeling_function()
        def but_0(x):
            pat="{action} but {negative_precondition}"
            if SnorkelUtil.pattern_exists(pat,x.text):
                return SnorkelUtil.DISABLING
            else:
                return SnorkelUtil.ABSTAIN
              
        @labeling_function()
        def makes_possible_1(x):
            pat="{precondition} makes {action} possible."
            if SnorkelUtil.pattern_exists(pat,x.text):
                return SnorkelUtil.ENABLING
            else:
                return SnorkelUtil.ABSTAIN
            
        @labeling_function()
        def to_understand_event_1(x):
            pat = r'To understand the event "{event}", it is important to know that {precondition}.'
            if SnorkelUtil.pattern_exists(pat,x.text):
                return SnorkelUtil.ENABLING
            else:
                return SnorkelUtil.ABSTAIN
          
            
        @labeling_function()
        def statement_is_true_1(x):
            pat = r'The statement "{event}" is true because {precondition}.'
            if SnorkelUtil.pattern_exists(pat,x.text):
                return SnorkelUtil.ENABLING
            else:
                return SnorkelUtil.ABSTAIN
            
        # enabling_dict={}
        # disabling_dict={}

        
        self.disabling_dict={
            'but' : "{action} but {negative_precondition}",
            'unless' : "{action} unless {precondition}",
            'if' : "{action} if not {precondition}",
            }
        
        
        self.enabling_dict={
            'makes possible' : "{precondition} makes {action} possible.",
            'to understand event' : r'To understand the event "{event}", it is important to know that {precondition}.',
            'statement is true' : r'The statement "{event}" is true because {precondition}.',
            'if' : "{action} if {precondition}",
            }
        
        for p_conj in self.pos_conj:
            self.lfs.append(self.make_keyword_lf(p_conj,SnorkelUtil.ENABLING))
            self.enabling_dict[p_conj]="{action} " +  p_conj + " {precondition}"
        
        self.lfs.extend([makes_possible_1, to_understand_event_1, statement_is_true_1])
            
        for n_conj in self.neg_conj:
           self.lfs.append(self.make_keyword_lf(n_conj,SnorkelUtil.DISABLING)) 
           self.disabling_dict[n_conj]="{action} " +  n_conj + " {precondition}"
            
        
        self.lfs.extend([unless_0, but_0, if_0])
        
        applier = PandasLFApplier(self.lfs)
        # global L_omcs
        self.L = applier.apply(df)
        
        print(LFAnalysis(self.L, self.lfs).lf_summary())
        # print(type(LFAnalysis(L_omcs, lfs).lf_summary()))
        
        self.LFA_df=LFAnalysis(self.L, self.lfs).lf_summary().copy()
        
        
     

    def get_L_matrix(self):
        return self.L, self.LFA_df

       
    @staticmethod
    def keyword_lookup(x, keyword, label):
        pat="{action} " +  keyword + " {precondition}."
        if SnorkelUtil.pattern_exists(pat,x.text):
            return label
        else:
            return SnorkelUtil.ABSTAIN
        
    @staticmethod
    def make_keyword_lf(keyword, label):
        lf_name=keyword.replace(" ","_") + f"_{label}"
        return LabelingFunction(
            name=lf_name,
            f=SnorkelUtil.keyword_lookup,
            resources=dict(keyword=keyword, label=label),
        )    
 
        
    @staticmethod
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
    
    
    @staticmethod
    def get_precondition_action(pattern,line):
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
            
            action=""
            precondition=""
            
            
            if 'negative_precondition' in pattern_keys:
                precondition=match_dict['negative_precondition']
            else:
                precondition=match_dict['precondition']
                
            if 'event' in pattern_keys:
                action=match_dict['event']
            else:
                action=match_dict['action']
    
            # if 'negative_action' in pattern_keys:
            #     action=match_dict['negative_action']
            # else:
            #     action=match_dict['action']
                
            
            return precondition, action
        
    @staticmethod  
    def returnExamples(L, LFA_df, df, N=100):
        lfs_names=list(LFA_df.index)
        df_data=None
        examples_df=pd.DataFrame()
        
        for index,row in LFA_df.iterrows():
            s_no=int(row['j'])
            label=int(index[-1])
    
            pat_matches=L[:, s_no] == label
            match_count=sum(bool(x) for x in pat_matches)
            tmp_list=list(df.iloc[L[:, s_no] == label].sample(min(match_count,N), random_state=1)['text'])
            if len(tmp_list)<N:
                tmp_list += [0] * (N - len(tmp_list))
            examples_df[str(index)]=tmp_list
        return examples_df
          
    
    #Adds Action, Precondtion columns to df.
    # @staticmethod  
    def addActionPrecondition(self, L, LFA_df, df):
        actions=[]
        preconditions=[]
        lfs_names=list(LFA_df.index)
        for index,row in df.iterrows():
            valid_positions=L[index,:] > -1
            # position = np.argmax(L[index,:] > -1)
            action=-1
            precondition=-1
            label=row["label"]
            
            if not(np.any(valid_positions)) or label==SnorkelUtil.ABSTAIN:
                action=-1
                precondition=-1
            else:
                if label==SnorkelUtil.ENABLING:
                    position = np.argmax(L[index,:] == SnorkelUtil.ENABLING)
                    conj=lfs_names[position][:-2].replace("_"," ")
                    pat=self.enabling_dict[conj]
                elif label==SnorkelUtil.DISABLING:
                    position = np.argmax(L[index,:] == SnorkelUtil.DISABLING)
                    conj=lfs_names[position][:-2].replace("_"," ")
                    pat=self.disabling_dict[conj]
                # else:                                                       #AMBIGUOUS PATTERN
                #     pat="{precondition} (?:so|hence|consequently) {action}."
                # position = np.argmax(L[index,:] > -1)
                # conj=lfs_names[position][:-2].replace("_"," ")
                # # print(conj)
                # if conj=="ambiguous pat":
                #     conj="(?:so|hence|consequently)"
                # pat="{action} " +  conj + " {precondition}."
                # if conj=="makespossible":
                #     pat="{precondition} makes {action} possible."
                    
                try:
                    precondition, action= SnorkelUtil.get_precondition_action(pat,row['text'])
                except Exception as e:
                    print(e)
                    print("pattern="+pat)
                    print("text="+row['text'])
            actions.append(action)
            preconditions.append(precondition)
        print("DF len="+str(len(df)))
        print("Actions len="+str(len(actions)))
        df['Action']=actions
        df['Precondition']=preconditions
        return df
    




