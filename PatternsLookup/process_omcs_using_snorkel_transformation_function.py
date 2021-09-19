# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:30:05 2021

@author: Dell
"""
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
from snorkel.augmentation import transformation_function
from snorkel.augmentation import PandasTFApplier

import nltk
from nltk.corpus import wordnet as wn

from snorkel.augmentation import RandomPolicy

nltk.download("wordnet")

from snorkel.preprocess.nlp import SpacyPreprocessor

spacy = SpacyPreprocessor(text_field="text", doc_field="doc", memoize=True)


import logging
logger = logging.getLogger(__name__)




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


ABSTAIN = -1
DISABLING = 0
ENABLING = 1
AMBIGUOUS=2


def pattern_exists(pattern,line):
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
        if 'negative_precondition' in pattern_keys:
                    if not(any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS])):
                        return False
    if len(m_list)>0:
        return True
    return False


def keyword_lookup(x, keyword, label):
    pat="{action} " +  keyword + " {precondition}."
    if pattern_exists(pat,x.text):
        return label
    else:
        return ABSTAIN


def make_keyword_lf(keyword, label):
    lf_name=keyword.replace(" ","_") + f"_{label}"
    return LabelingFunction(
        name=lf_name,
        f=keyword_lookup,
        resources=dict(keyword=keyword, label=label),
    )


lfs=[]

pos_conj = {'only if', 'contingent upon', 'given', 'if',"in case", "in the case that", "in the event", "on condition", "on the assumption",
            "on these terms", "subject to", "supposing", "with the proviso"}

neg_conj = {"but", "except", "except for", "excepting that", "if not", "lest", "saving", "without"}


for p_conj in pos_conj:
    lfs.append(make_keyword_lf(p_conj,ENABLING))
    
    
for n_conj in neg_conj:
   lfs.append(make_keyword_lf(n_conj,DISABLING))   
    


@labeling_function()
def disabling1(x):
    for pat in SINGLE_SENTENCE_DISABLING_PATTERNS1:
        if pattern_exists(pat,x.text):
            return DISABLING
    return ABSTAIN


              
@labeling_function()
def enabling_makespossible(x):
    pat="{precondition} makes {action} possible."
    if pattern_exists(pat,x.text):
        return ENABLING
    else:
        return ABSTAIN
    
    
@labeling_function()
def ambiguous_pat(x):
    pat="{precondition} (?:so|hence|consequently) {action}."
    if pattern_exists(pat,x.text):
        return AMBIGUOUS
    else:
        return ABSTAIN


lfs.extend([disabling1, enabling_makespossible, ambiguous_pat])

def replace_token(spacy_doc, idx, replacement):
    """Replace token in position idx with replacement."""
    return " ".join([spacy_doc[:idx].text, replacement, spacy_doc[1 + idx :].text])

def get_synonym(word, pos=None):
    """Get synonym for word given its part-of-speech (pos)."""
    synsets = wn.synsets(word)
    # Return None if wordnet has no synsets (synonym sets) for this word and pos.
    if synsets:
        words = [lemma.name() for lemma in synsets[0].lemmas()]
        if words[0].lower() != word.lower():  # Skip if synonym is same as word.
            # Multi word synonyms in wordnet use '_' as a separator e.g. reckon_with. Replace it with space.
            return words[0].replace("_", " ")


@transformation_function(pre=[spacy])
def replace_conj_with_synonym(x):
    # Get indices of verb tokens in sentence.
    conj_idxs = [i for i, token in enumerate(x.doc) if (token.pos_ == "SCONJ" or token.pos_ == "CCONJ")]
    if conj_idxs:
        # Pick random verb idx to replace.
        idx = np.random.choice(conj_idxs)
        synonym = get_synonym(x.doc[idx].text)
        # If there's a valid conj synonym, replace it. Otherwise, return None.
        if synonym:
            x.text = replace_token(x.doc, idx, synonym)
            return x



tfs = [replace_conj_with_synonym]



random_policy = RandomPolicy(len(tfs))

@hydra.main(config_path="../Configs", config_name="snorkel_config")
def main(config: omegaconf.dictconfig.DictConfig):
    omcs_df = pd.read_csv(config.corpus_path, sep="\t", error_bad_lines=False)
    omcs_df['text'] = omcs_df['text'].astype(str)
    
    tf_applier = PandasTFApplier(tfs,random_policy)
    omcs_df_augmented = tf_applier.apply(omcs_df)
    
    applier = PandasLFApplier(lfs)
    
    L_omcs = applier.apply(omcs_df)
    L_omcs_augmented = applier.apply(omcs_df_augmented)
    
    print(LFAnalysis(L_omcs_augmented, lfs).lf_summary())
    
    # Train the label model and compute the training labels
    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L_omcs_augmented, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    
    omcs_df["label"] = label_model.predict(L=L_omcs, tie_break_policy="abstain")
    
    omcs_df = omcs_df[omcs_df.label != ABSTAIN]
    
    print(config.output_name)
    omcs_df.to_csv(config.output_name)
    
    count = omcs_df["label"].value_counts()
    print("Label  Count")
    print(count)
    


if __name__ == '__main__':
    main()





