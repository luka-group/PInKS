import re
from typing import NoReturn
import logging

import IPython
import omegaconf
import nltk
import numpy as np
import pandas as pd
import tqdm
from snorkel.labeling import LFAnalysis
from snorkel.labeling import LabelingFunction
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import labeling_function
from snorkel.labeling.model import LabelModel

from Patterns import PatternUtils

nltk.download("wordnet")


logger = logging.getLogger(__name__)


class SnorkelUtil:
    ABSTAIN = -1
    DISABLING = 0
    ENABLING = 1

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
        # Not in the labeled data
        "with the proviso": 0.0,
        "on these terms": 0.0,
        "only if": 0.0,
        "make possible": 0.0,
        "without": 0.0,
        "excepting that": 0.0,
    }

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        self.config = config
        self.L = None
        self.LFA_df = None

        self.np_lf_recalls = None

    def apply_labeling_functions(self, df: pd.DataFrame) -> NoReturn:

        self.populate_labeling_functions_list()
        assert self.np_lf_recalls is not None

        applier = PandasLFApplier(self.lfs)
        # global L_omcs
        self.L = applier.apply(df)
        logger.info(LFAnalysis(self.L, self.lfs).lf_summary())
        self.LFA_df = LFAnalysis(self.L, self.lfs).lf_summary().copy()

        label_model = LabelModel(cardinality=3, verbose=True)
        label_model.fit(self.L, n_epochs=self.config.snorkel_epochs, log_freq=50, seed=123)
        df["label"] = label_model.predict(L=self.L, tie_break_policy="abstain")

    def return_examples(self, df: pd.DataFrame, num: int = 100) -> pd.DataFrame:
        lfs_names = list(self.LFA_df.index)
        df_data = None
        examples_df = pd.DataFrame()

        for index, row in self.LFA_df.iterrows():
            s_no = int(row['j'])
            label = int(index[-1])

            pat_matches = self.L[:, s_no] == label
            match_count = sum(bool(x) for x in pat_matches)
            tmp_list = list(df.iloc[self.L[:, s_no] == label].sample(min(match_count, num), random_state=1)['text'])
            if len(tmp_list) < num:
                tmp_list += [0] * (num - len(tmp_list))
            examples_df[str(index)] = tmp_list
        return examples_df

    def add_action_precondition(self, df: pd.DataFrame):
        actions = []
        preconditions = []
        lfs_names = list(self.LFA_df.index)

        pattern_lookup = {
            self.ENABLING: self.enabling_dict,
            self.DISABLING: self.disabling_dict,
        }
        for index, row in tqdm.tqdm(df.iterrows(), desc='Extracting Action/Precondition'):

            label = row["label"]
            if label == self.ABSTAIN or (not np.any(self.L[index, :] == label)):
                actions.append(-1)
                preconditions.append(-1)
                continue

            try:
                position = np.argmax(np.multiply((self.L[index, :] == label).astype(float), self.np_lf_recalls))
                conj = lfs_names[position].replace("_", " ")
                pat = pattern_lookup[label][conj]
            except Exception as e:
                logger.error(f'{e}')
                IPython.embed()
                exit()

            try:
                precondition, action = self.get_precondition_action(pat, row['text'])
                actions.append(action)
                preconditions.append(precondition)
            except Exception as e:
                text = row['text']
                logger.error(f"pattern={pat}, text={text}, e={e}")
                actions.append(-1)
                preconditions.append(-1)

        logger.info("DF len=" + str(len(df)))
        logger.info("Actions len=" + str(len(actions)))
        df['action'] = actions
        df['precondition'] = preconditions

    def populate_labeling_functions_list(self) -> NoReturn:
        pos_conj = {'only if', 'contingent upon',  "in case", "in the case that", "in the event",
                    "on condition", "on the assumption",
                    "on these terms", "supposing", "with the proviso"}
        neg_conj = {"except", "except for", "excepting that", "lest", "without", "unless"}
        self.disabling_dict = {
            'but': "{action} but {negative_precondition}",
            # 'unless': "{action} unless {precondition}",
            'if not': "{action} if not {precondition}",
        }
        self.enabling_dict = {
            'makes possible': "{precondition} makes {action} possible.",
            'to understand event': r'To understand the event "{event}", it is important to know that {precondition}.',
            'statement is true': r'The statement "{event}" is true because {precondition}.',
            'if': "{action} if {precondition}",
        }
        self.lfs = []
        lf_recalls = []

        def gimme_recall(conj):
            try:
                return self.LF_RECALS[conj]
            except KeyError:
                logger.error(f'Add {conj}')
                return float(self.config.lf_recall_threshold)

        for p_conj in pos_conj:
            recall = gimme_recall(p_conj)
            if recall < float(self.config.lf_recall_threshold):
                continue

            assert p_conj not in self.enabling_dict, p_conj

            self.lfs.append(self.make_keyword_lf(p_conj, SnorkelUtil.ENABLING))
            self.enabling_dict[p_conj] = "{action} " + p_conj + " {precondition}"
            lf_recalls.append(recall)

        for p_conj, lf in zip(
                ['make possible', 'to understand event', 'statement is true', 'if'],
                [self.makes_possible_1, self.to_understand_event_1, self.statement_is_true_1, self.if_1]):
            recall = gimme_recall(p_conj)
            if recall < float(self.config.lf_recall_threshold):
                continue
            self.lfs.append(lf)
            lf_recalls.append(recall)

        # self.lfs.extend([self.makes_possible_1, self.to_understand_event_1, self.statement_is_true_1])
        for n_conj in neg_conj:
            recall = gimme_recall(n_conj)
            if recall < float(self.config.lf_recall_threshold):
                continue

            assert n_conj not in self.disabling_dict, n_conj
            self.lfs.append(self.make_keyword_lf(n_conj, SnorkelUtil.DISABLING))
            self.disabling_dict[n_conj] = "{action} " + n_conj + " {precondition}"

            lf_recalls.append(recall)

        for n_conj, lf in zip(['but', 'if not'], [self.but_0, self.if_not_0]):
            recall = gimme_recall(n_conj)
            if recall < float(self.config.lf_recall_threshold):
                continue

            self.lfs.append(lf)
            lf_recalls.append(recall)

        self.np_lf_recalls = np.array(lf_recalls)

    @staticmethod
    @labeling_function(name='if')
    def if_1(x):
        if_not_pat = "{action} if not {precondition}"
        if_pat = "{action} if {precondition}"
        if SnorkelUtil.pattern_exists(if_pat, x.text) and not(SnorkelUtil.pattern_exists(if_not_pat, x.text)):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function(name='if not')
    def if_not_0(x):
        pat = "{action} if not {precondition}"
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.DISABLING
        else:
            return SnorkelUtil.ABSTAIN

    # @staticmethod
    # @labeling_function()
    # def unless_0(x):
    #     for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS1:
    #         if SnorkelUtil.pattern_exists(pat, x.text):
    #             return SnorkelUtil.DISABLING
    #     return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function(name='but')
    def but_0(x):
        pat = "{action} but {negative_precondition}"
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.DISABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function(name='make possible')
    def makes_possible_1(x):
        pat = "{precondition} makes {action} possible."
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function(name='to understand event')
    def to_understand_event_1(x):
        pat = r'To understand the event "{event}", it is important to know that {precondition}.'
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function(name='statement is true')
    def statement_is_true_1(x):
        pat = r'The statement "{event}" is true because {precondition}.'
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    def keyword_lookup(x, keyword, label):
        pat = "{action} " + keyword + " {precondition}."
        if SnorkelUtil.pattern_exists(pat, x.text):
            return label
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    def make_keyword_lf(keyword, label):
        lf_name = keyword.replace(" ", "_")  # + f"_{label}"
        return LabelingFunction(
            name=lf_name,
            f=SnorkelUtil.keyword_lookup,
            resources=dict(keyword=keyword, label=label),
        )

    @staticmethod
    def pattern_exists(pattern, line):
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
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

    @staticmethod
    def get_precondition_action(pattern, line):
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

