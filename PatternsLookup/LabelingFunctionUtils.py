import abc
import re
from typing import NoReturn, Callable, Dict, Tuple, List
import logging

import IPython
import functools
import omegaconf
import nltk
import numpy as np
import pandas as pd
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel

from Patterns import PatternUtils

nltk.download("wordnet")


logger = logging.getLogger(__name__)


class LF(abc.ABC):
    ABSTAIN = -1
    DISABLING = 0
    ENABLING = 1

    def __init__(self):
        super(LF, self).__init__()
        self.NAME = 'BaseLF'

    @abc.abstractmethod
    def get_labeling_func(self) -> Callable[[str], int]:
        pass

    @abc.abstractmethod
    def get_splitter_func(self) -> Callable[[str], Dict[str, str]]:
        pass

    @abc.abstractmethod
    def get_split_pattern(self) -> str:
        pass

    @staticmethod
    def pattern_exists(line: str, pattern: str) -> bool:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(**replacements)
        m_list = re.findall(regex_pattern, line)

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
        if len(m_list) > 0:
            return True
        return False

    @staticmethod
    def get_precondition_action(line: str, pattern: str) -> Tuple[str, str]:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(**replacements)
        m_list = re.findall(regex_pattern, line)
        for m in m_list:
            match_full_sent = line
            for sent in line:
                if all([ps in sent for ps in m]):
                    match_full_sent = sent
            match_dict = dict(zip(pattern_keys, m))

            action = ""
            precondition = ""

            if 'negative_precondition' in pattern_keys:
                precondition = match_dict['negative_precondition']
            else:
                precondition = match_dict['precondition']

            if 'event' in pattern_keys:
                action = match_dict['event']
            else:
                action = match_dict['action']

            # if 'negative_action' in pattern_keys:
            #     action=match_dict['negative_action']
            # else:
            #     action=match_dict['action']

            return precondition, action


class MakePossibleLF(LF):

    def __init__(self):
        super(LF, self).__init__()
        self.NAME = "make possible"

    def get_splitter_func(self) -> Callable[[str], Dict[str, str]]:
        return functools.partial(self.get_precondition_action, pattern=self.get_split_pattern())

    def get_split_pattern(self) -> str:
        return '{precondition} makes {action} possible.'

    def get_labeling_func(self) -> Callable[[str], int]:
        lf_pattern = self.get_split_pattern()

        @labeling_function()
        def mylf(x, pat):
            if LF.pattern_exists(pat, x.text):
                return LF.ENABLING
            else:
                return LF.ABSTAIN

        return functools.partial(mylf, pat=lf_pattern)


class ToUnderstandEventLF(MakePossibleLF):

    def __init__(self):
        super(LF, self).__init__()
        self.NAME = "to understand event"

    def get_split_pattern(self) -> str:
        return r'To understand the event "{event}", it is important to know that {precondition}.'


class StatementIsTrue1LF(MakePossibleLF):

    def __init__(self):
        super(LF, self).__init__()
        self.NAME = "statement is true"

    def get_split_pattern(self) -> str:
        return r'The statement "{event}" is true because {precondition}.'


class But0LF(MakePossibleLF):

    def __init__(self):
        super(LF, self).__init__()
        self.NAME = "but 0"

    def get_split_pattern(self) -> str:
        return r'{action} but {negative_precondition}'

    def get_labeling_func(self) -> Callable[[str], int]:
        lf_pattern = self.get_split_pattern()

        def mylf(x, pat):
            if LF.pattern_exists(pat, x.text):
                return LF.DISABLING
            else:
                return LF.ABSTAIN

        return LabelingFunction(
            name=self.NAME,
            f=mylf,
            resources=dict(pat=lf_pattern),
        )


class If0LF(LF):
    def __init__(self):
        super(LF, self).__init__()
        self.NAME = "if not"

    def get_split_pattern(self) -> str:
        return '{action} if not {precondition}'

    def get_labeling_func(self) -> Callable[[str], int]:
        lf_pattern = self.get_split_pattern()

        def mylf(x, pat):
            if SnorkelUtil.pattern_exists(pat, x.text):
                return LF.DISABLING
            elif SnorkelUtil.pattern_exists("{action} if {precondition}", x.text):
                return LF.ENABLING
            else:
                return LF.ABSTAIN

        return LabelingFunction(
            name=self.NAME,
            f=mylf,
            resources=dict(pat=lf_pattern),
        )
        # return functools.partial(mylf, pat=lf_pattern)


class GenericEnablingConj(MakePossibleLF):
    def __init__(self, conj: str, label: int):
        super(LF, self).__init__()
        self.NAME = conj
        self.CONJ = conj
        self.LABEL = label

    def get_split_pattern(self) -> str:
        return "{action} " + self.CONJ + " {precondition}."

    @staticmethod
    def make_keyword_lf(keyword, label):
        lf_name = keyword.replace(" ", "_") + f"_{label}"
        return LabelingFunction(
            name=lf_name,
            f=GenericEnablingConj.keyword_lookup,
            resources=dict(keyword=keyword, label=label),
        )

    @staticmethod
    def keyword_lookup(x, keyword, label):
        pat = "{action} " + keyword + " {precondition}."
        if LF.pattern_exists(pat, x.text):
            return label
        else:
            return LF.ABSTAIN


class LabelingFunctionUtils:
    LF_RECALS = {
        "but": 0.17,
        "contingent upon": 0.6,
        "except": 0.7,
        "except for": 0.57,
        "if": 0.52,
        "if not": .97,
        "in case": .75,
        "in the case that": .30,
        "in the event": .3,
        "lest": .06,
        "makes possible": .81,
        "on condition": .6,
        "on the assumption": 0.44,
        "statement is true": 1.0,
        "supposing": .07,
        "to understand event": .87,
        "unless": 1.0,
        "with the proviso": 0.0,
    }

    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_labeling_functions(config) -> List["LF"]:
        pos_conj = {'only if', 'contingent upon', 'if', "in case", "in the case that", "in the event",
                    "on condition", "on the assumption",
                    "on these terms", "supposing", "with the proviso"}
        neg_conj = {"except", "except for", "excepting that", "if not", "lest", "without", "unless"}
