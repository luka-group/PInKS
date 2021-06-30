import json
import os
import pathlib
import re
from typing import List, Dict, Generator
import subprocess
import IPython
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging
import logging
import pandas as pd

logger = logging.getLogger(__name__)

FACT_REGEX = r'([a-zA-Z0-9_\-\\\/\+\* \'’%]{10,})'


class PatternUtils:
    @staticmethod
    def get_roles_from_desc(desc: str, prefix: str) -> Dict:
        if len(desc) == 0:
            args_dict = {}
        else:
            args_dict = dict(re.findall(r'\[([^:]+): ([^\]]+)]', desc))
        return {
            # f'{prefix}_verbs': args_dict.get('V', ''),
            **{f'{prefix}_{k}': args_dict.get(k, '') for k in args_dict.keys()},
            # **{f'{prefix}_arg{p}': args_dict.get(f'ARG{p}', '') for p in ['M-PRD']},
        }

    @staticmethod
    def clean_srl(d: Dict, prefix: str = '') -> List[Dict]:
        if len(d['verbs']) == 0:
            # print('unable to parse {}'.format(d['words']))
            return [PatternUtils.get_roles_from_desc('', prefix=prefix)]
        # elif len(d['verbs']) == 1:
        #     return PatternUtils.get_roles_from_desc(d['verbs'][0]['description'], prefix=prefix)

        filtered_d = sorted(list(filter(
            lambda v: v['verb'] not in [],
            filter(
                lambda v: 'description' in v.keys() and 'tags' in v.keys(),
                d['verbs']
            ))),
            key=lambda v: v['tags'].count('O')
        )

        # return PatternUtils.get_roles_from_desc(filtered_d[0]['description'], prefix=prefix)
        return [PatternUtils.get_roles_from_desc(d['description'], prefix=prefix) for d in filtered_d]

    @staticmethod
    def split_sentences(matches: List[str]):
        for m in matches:
            nltk.sent_tokenize(m)

    @staticmethod
    def check_pattern_in_file_grep(pattern: str, base_path: str, files_pattern: str,
                                   do_srl: bool = False, label: str = '') \
            -> List[Dict[str, str]]:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(**replacements)

        with open(pathlib.Path(os.getcwd()) / f'temp_bash.sh', 'w') as fp:
            fp.write(" ".join([
                'grep', '-iE', fr'"{regex_pattern}"',
                os.path.join(base_path, files_pattern),
                '-a', '> ',
                os.path.join(os.getcwd(), 'OUT.tmp')
                # '> temp_bash_output'
            ]))

        try:
            subprocess.run(['sh', 'temp_bash.sh'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f'Failed grep (code {e}). Search using iterative method.')
            return PatternUtils.check_pattern_in_files(pattern, base_path, files_pattern, do_srl, label)

        return PatternUtils.check_pattern_in_files(
            pattern, base_path=os.getcwd(), files_pattern='OUT.tmp',
            do_srl=do_srl, label=label
        )
    
    
    @staticmethod
    def check_pattern_in_files_omcs(pattern: str, base_path: str, files_pattern: str,
                               do_srl: bool = False, label: str = '') \
            -> List[Dict[str, str]]:
        logger.info(f'Checking pattern ({pattern}) in files {files_pattern}')

        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)    #Extracts string written between {}
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}  #dictionary to replace action/precond with FACT_REGEX
        regex_pattern = pattern.format(     #Replace the pattern with regex in place of {action} and {precondition}.
            **replacements
        )

        for k in ['any_word', 'ENB_CONJ']:
            try:
                pattern_keys.remove(k)
            except ValueError:
                pass

        all_matches = []
        for f in tqdm(pathlib.Path(base_path).iterdir(), desc='files'):
            if files_pattern is not None and files_pattern not in str(f):
                continue
            omcs_df = pd.read_csv(f, sep="\t", error_bad_lines=False)
            for ind in omcs_df.index:
                # print(df['Name'][ind], df['Stream'][ind])
                fix_d = {
                    'pattern': pattern,
                    'label': label
                }
                for new_match in PatternUtils.find_matches_in_line_gigaword(line=str(omcs_df['text'][ind]), regex_pattern=regex_pattern,
                                                                   pattern_keys=pattern_keys):
                    all_matches.append({**new_match, **fix_d})

        if do_srl:
            PatternUtils.process_matches_with_srl(all_matches, pattern_keys)
        else:
            logger.warning(f'Skipping SRL')

        return all_matches


    @staticmethod
    def check_pattern_in_files_gigaword(pattern: str, base_path: str, files_pattern: str,
                               do_srl: bool = False, label: str = '') \
            -> List[Dict[str, str]]:
        logger.info(f'Checking pattern ({pattern}) in files {files_pattern}')

        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)    #Extracts string written between {}
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}  #dictionary to replace action/precond with FACT_REGEX
        regex_pattern = pattern.format(     #Replace the pattern with regex in place of {action} and {precondition}.
            **replacements
        )

        for k in ['any_word', 'ENB_CONJ']:
            try:
                pattern_keys.remove(k)
            except ValueError:
                pass

        all_matches = []
        for f in tqdm(pathlib.Path(base_path).iterdir(), desc='files'):
            if files_pattern is not None and files_pattern not in str(f):
                continue
            with open(f, 'r') as fp:
                for line in tqdm(fp, desc='lines'):
                    # print(line)
                    fix_d = {
                        'pattern': pattern,
                        'label': label
                    }
                    for new_match in PatternUtils.find_matches_in_line_gigaword(line=line, regex_pattern=regex_pattern,
                                                                       pattern_keys=pattern_keys):
                        all_matches.append({**new_match, **fix_d})

        if do_srl:
            PatternUtils.process_matches_with_srl(all_matches, pattern_keys)
        else:
            logger.warning(f'Skipping SRL')

        return all_matches

    @staticmethod
    def process_matches_with_srl(all_matches, pattern_keys):
        parser = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            import_plugins=True
        )
        # IPython.embed()
        for match in tqdm(all_matches, desc='SRL'):
            for k in pattern_keys:
                match[f'parsed_{k}'] = PatternUtils.clean_srl(parser.predict(sentence=match.get(k, '')))

    @staticmethod
    def find_matches_in_line(line: str, regex_pattern: str, pattern_keys: List[str]) \
            -> Generator[Dict[str, str], None, None]:
        m_list = re.findall(regex_pattern, line)
        jline = json.loads(line)
        for m in m_list:
            match_full_sent = jline['sources']
            for sent in jline['sources']:
                if all([ps in sent for ps in m]):
                    match_full_sent = sent

            match_dict = dict(zip(pattern_keys, m))

            flags = []
            if 'negative_precondition' in pattern_keys:
                if any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS]):
                    flags.append('NEG_TO_POS')
                    match_full_sent = PatternUtils.make_sentence_positive(match_full_sent)
                    match_dict['precondition'] = PatternUtils.make_sentence_positive(match_dict['negative_precondition'])
                    match_dict.pop('negative_precondition')
                else:
                    continue

            yield {
                'line': match_full_sent,
                **jline,
                **match_dict,
                'flags': flags,
            }
            
    @staticmethod
    def find_matches_in_line_gigaword(line: str, regex_pattern: str, pattern_keys: List[str]) \
            -> Generator[Dict[str, str], None, None]:
        m_list = re.findall(regex_pattern, line)
        # jline = json.loads(line)
        for m in m_list:
            match_full_sent = line
            for sent in line:
                if all([ps in sent for ps in m]):
                    match_full_sent = sent

            match_dict = dict(zip(pattern_keys, m))

            flags = []
            if 'negative_precondition' in pattern_keys:
                if any([nw in match_dict['negative_precondition'] for nw in PatternUtils.NEGATIVE_WORDS]):
                    flags.append('NEG_TO_POS')
                    match_full_sent = PatternUtils.make_sentence_positive(match_full_sent)
                    match_dict['precondition'] = PatternUtils.make_sentence_positive(match_dict['negative_precondition'])
                    match_dict.pop('negative_precondition')
                else:
                    continue

            yield {
                'line': match_full_sent,
                'og_line':line,
                **match_dict,
                'flags': flags,
            }

    @staticmethod
    def make_sentence_positive(sent: str) -> str:
        negative_dict = {
            ' not ': ' ',
            ' cannot ': ' can ',
            'n\'t ': ' ',
            ' don\\u2019t ': ' do ',
            ' doesn\\u2019t ': ' does ',
        }
        assert len(negative_dict) == len(PatternUtils.NEGATIVE_WORDS)
        for kn, pv in negative_dict.items():
            if kn in sent:
                sent = sent.replace(kn, pv)

        return sent

    #################################################################################
    FACT_REGEX = FACT_REGEX
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

    NEGATIVE_WORDS = [
        ' not ',
        ' cannot ',
        'n\'t ',
        ' don\\u2019t ',
        ' doesn\\u2019t ',
    ]


    # PRECONDITION_REGEX = r'([\w\-\\\/\+\* ,\']+)'
    SINGLE_SENTENCE_DISABLING_PATTERNS = [
        r"^{action} unless {precondition}\.",
        r"\. {action} unless {precondition}\.",
        r"^{any_word} unless {precondition}, {action}\.",
        r"^{any_word} unless {precondition}, {action}\.",
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
