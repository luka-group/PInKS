# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 23:04:14 2021

@author: Dell
"""
import re, spacy
from skweak import heuristics, gazetteers, aggregation, utils
import json


nlp = spacy.load("en_core_web_sm")



FACT_REGEX = r'([a-zA-Z0-9_\-\\\/\+\* \'’%]{10,})'

REPLACEMENT_REGEX = {
        'action': FACT_REGEX,
        'precondition': FACT_REGEX,
        'negative_precondition': FACT_REGEX,
        'precondition_action': FACT_REGEX,
        'any_word': r'[^ \[]{,10}',
        'ENB_CONJ': r'(?:so|hence|consequently|thus|therefore|'
                    r'as a result|thus|accordingly|because of that|'
                    r'as a consequence|as a result)',
    }

# pattern = "{action} unless {precondition}"




SINGLE_SENTENCE_DISABLING_PATTERNS1 = [
    r"^{action} unless {precondition}\.",
    r"\. {action} unless {precondition}\.",
    r"^{any_word} unless {precondition}, {action}\.",
    r"^{any_word} unless {precondition}, {action}\.",
]

SINGLE_SENTENCE_DISABLING_PATTERNS2 = [
    r"{negative_precondition} (?:so|hence|consequently) {action}\.",
]


nlp.max_length = 34664450

#Add path to corpus
f = open(r"C:\Users\Dell\Desktop\Piyush\USC-CQ\Gigaword10k_msft\org_data\test_src\test.src.txt","r")
doc=nlp(f.read())


def pattern_exists(pattern,sent):
    pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
    replacements = {k: REPLACEMENT_REGEX[k] for k in pattern_keys}    
    regex_pattern = pattern.format(**replacements)
    m_list = re.findall(regex_pattern, sent)
    if len(m_list)>0:
        return True
    return False
    
    
    
def single_sent_disabling_pat1(doc):
  for sent in doc.sents:
      for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
          if pattern_exists(pat,sent.text):
              yield sent.start,sent.end,"DISABLING1"

lf1= heuristics.FunctionAnnotator("disabling1", single_sent_disabling_pat1)



def single_sent_disabling_pat2(doc):
  for sent in doc.sents:
      for pat in SINGLE_SENTENCE_DISABLING_PATTERNS2:
          if pattern_exists(pat,sent.text):
              yield sent.start,sent.end,"DISABLING2"
              
lf2= heuristics.FunctionAnnotator("disabling2", single_sent_disabling_pat2)


# apply the labelling functions
doc = lf2(lf1(doc))

# and aggregate them
hmm = aggregation.HMM("hmm", ["DISABLING1","DISABLING2"])
hmm.fit_and_aggregate([doc])

# we can then visualise the final result (in Jupyter)
utils.display_entities(doc, "hmm")


skweak_hits=doc.spans["hmm"]

hits_list=[]
for hit in skweak_hits:
    hits_list.append(hit.text)

skweak_matches={}
skweak_matches["matches"]=list(hits_list)


with open('../Outputs/skweak_matches_test.json', 'w') as fp:
    json.dump(skweak_matches, fp)










