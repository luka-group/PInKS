# -*- coding: utf-8 -*-
"""
Created on Sat Sep 11 19:43:11 2021

@author: Dell
"""
import re

import nltk
import pandas as pd
from tqdm import tqdm

from Patterns import PatternUtils

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


def pattern_exists(pattern, line):
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
    if len(m_list) > 0:
        return True
    return False


pos_conj = {'only if', 'contingent upon', "in case", "in the case that", "in the event", "on condition",
            "on the assumption",
            "on these terms", "supposing", "with the proviso"}

neg_conj = {"except", "except for", "excepting that", "lest"}

enabling_dict = {}
disabling_dict = {}

disabling_dict = {
    'but': "{action} but {negative_precondition}",
    'unless': "{action} unless {precondition}",
    'if not': "{action} if not {precondition}",
}

enabling_dict = {
    'makes possible': "{precondition} makes {action} possible.",
    'to understand event': r'To understand the event "{event}", it is important to know that {precondition}.',
    'statement is true': r'The statement "{event}" is true because {precondition}.',
    'if': "{action} if {precondition}",
}

lfs = []

for n_conj in neg_conj:
    # lfs.append(make_keyword_lf(n_conj,DISABLING))
    disabling_dict[n_conj] = "{action} " + n_conj + " {precondition}."

for p_conj in pos_conj:
    # lfs.append(make_keyword_lf(p_conj,ENABLING))
    enabling_dict[p_conj] = "{action} " + p_conj + " {precondition}."

# lfs.extend([makes_possible_1, to_understand_event_1, statement_is_true_1])


# lf_pats=[]

# for key,val in disabling_dict.items():
#     lf_pats.append(val)

# for key,val in enabling_dict.items():
#     lf_pats.append(val)


lf_instances = {}

lf_instances.update(disabling_dict)
lf_instances.update(enabling_dict)

for key, val in lf_instances.items():
    lf_instances[key] = []

filtered_corpus_path = "/nas/home/pkhanna/CQplus/Outputs/filter_dataset/filtered_dataset.csv"

output_path = "/nas/home/pkhanna/CQplus/Outputs/LF_annotations.csv"


# @hydra.main(config_path="../Configs", config_name="annotation_creator")
# def main(config: omegaconf.dictconfig.DictConfig):
def main():
    filtered_df = pd.read_csv(filtered_corpus_path)

    for index, row in tqdm(filtered_df.iterrows()):
        text = row["text"]
        found = False
        for key, val in disabling_dict.items():
            if pattern_exists(val, text):
                lf_instances[key].append(text)
                found = True
                break
        if found:
            continue
        for key, val in enabling_dict.items():
            if pattern_exists(val, text):
                lf_instances[key].append(text)
                found = True
                break

    # lf_instances_df=pd.DataFrame.from_dict(lf_instances)

    lf_instances_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in lf_instances.items()]))

    print(lf_instances_df.count())

    # for key, val in lf_instances.items():
    #     label_row_name = key + " label"
    #     lf_instances_df[label_row_name] = ""

    # lf_instances_df = lf_instances_df.reindex(sorted(lf_instances_df.columns), axis=1)

    # lf_instances_df.head(100).to_csv(output_path)


if __name__ == '__main__':
    main()
