import re
from typing import NoReturn
import logging
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


# ABSTAIN = -1
# DISABLING = 0
# ENABLING = 1
# AMBIGUOUS=2


class SnorkelUtil:
    ABSTAIN = -1
    DISABLING = 0
    ENABLING = 1

    # AMBIGUOUS=2

    def __init__(self, config: omegaconf.dictconfig.DictConfig):
        self.config = config
        self._populate_labeling_functions_list()

    def _apply_labeling_functions(self, df: pd.DataFrame) -> NoReturn:
        applier = PandasLFApplier(self.lfs)
        # global L_omcs
        self.L = applier.apply(df)
        logger.info(LFAnalysis(self.L, self.lfs).lf_summary())
        self.LFA_df = LFAnalysis(self.L, self.lfs).lf_summary().copy()

        label_model = LabelModel(cardinality=3, verbose=True)
        label_model.fit(self.L, n_epochs=self.config.snorkel_epochs, log_freq=50, seed=123)
        df["label"] = label_model.predict(L=self.L, tie_break_policy="abstain")

    @staticmethod
    @labeling_function()
    def if_0(x):
        pat = "{action} if not {precondition}"
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.DISABLING
        elif SnorkelUtil.pattern_exists("{action} if {precondition}", x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function()
    def unless_0(x):
        for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS1:
            if SnorkelUtil.pattern_exists(pat, x.text):
                return SnorkelUtil.DISABLING
        return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function()
    def but_0(x):
        pat = "{action} but {negative_precondition}"
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.DISABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function()
    def makes_possible_1(x):
        pat = "{precondition} makes {action} possible."
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function()
    def to_understand_event_1(x):
        pat = r'To understand the event "{event}", it is important to know that {precondition}.'
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    @labeling_function()
    def statement_is_true_1(x):
        pat = r'The statement "{event}" is true because {precondition}.'
        if SnorkelUtil.pattern_exists(pat, x.text):
            return SnorkelUtil.ENABLING
        else:
            return SnorkelUtil.ABSTAIN

    def _populate_labeling_functions_list(self) -> NoReturn:
        pos_conj = {'only if', 'contingent upon', 'if', "in case", "in the case that", "in the event",
                         "on condition", "on the assumption",
                         "on these terms", "supposing", "with the proviso"}
        neg_conj = {"except", "except for", "excepting that", "if not", "lest", "without"}
        self.disabling_dict = {
            'but': "{action} but {negative_precondition}",
            'unless': "{action} unless {precondition}",
            'if': "{action} if not {precondition}",
        }
        self.enabling_dict = {
            'makes possible': "{precondition} makes {action} possible.",
            'to understand event': r'To understand the event "{event}", it is important to know that {precondition}.',
            'statement is true': r'The statement "{event}" is true because {precondition}.',
            'if': "{action} if {precondition}",
        }
        self.lfs = []
        for p_conj in pos_conj:
            self.lfs.append(self.make_keyword_lf(p_conj, SnorkelUtil.ENABLING))
            self.enabling_dict[p_conj] = "{action} " + p_conj + " {precondition}"
        self.lfs.extend([self.makes_possible_1, self.to_understand_event_1, self.statement_is_true_1])

        for n_conj in neg_conj:
            self.lfs.append(self.make_keyword_lf(n_conj, SnorkelUtil.DISABLING))
            self.disabling_dict[n_conj] = "{action} " + n_conj + " {precondition}"
        self.lfs.extend([self.unless_0, self.but_0, self.if_0])

    def get_L_matrix(self):
        return self.L, self.LFA_df

    @staticmethod
    def keyword_lookup(x, keyword, label):
        pat = "{action} " + keyword + " {precondition}."
        if SnorkelUtil.pattern_exists(pat, x.text):
            return label
        else:
            return SnorkelUtil.ABSTAIN

    @staticmethod
    def make_keyword_lf(keyword, label):
        lf_name = keyword.replace(" ", "_") + f"_{label}"
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
        for index, row in df.iterrows():
            valid_positions = self.L[index, :] > -1
            # position = np.argmax(L[index,:] > -1)
            action = -1
            precondition = -1
            label = row["label"]

            if not (np.any(valid_positions)) or label == SnorkelUtil.ABSTAIN:
                action = -1
                precondition = -1
            else:
                # to suppress the warning
                pat = ""
                if label == self.ENABLING:
                    position = np.argmax(L[index, :] == self.ENABLING)
                    conj = lfs_names[position][:-2].replace("_", " ")
                    pat = self.enabling_dict[conj]
                elif label == self.DISABLING:
                    position = np.argmax(L[index, :] == self.DISABLING)
                    conj = lfs_names[position][:-2].replace("_", " ")
                    pat = self.disabling_dict[conj]

                try:
                    precondition, action = self.get_precondition_action(pat, row['text'])
                except Exception as e:
                    logger.error(e)
                    logger.error("pattern=" + pat)
                    logger.error("text=" + row['text'])
            actions.append(action)
            preconditions.append(precondition)
        logger.info("DF len=" + str(len(df)))
        logger.info("Actions len=" + str(len(actions)))
        df['Action'] = actions
        df['Precondition'] = preconditions
        return df
