import os
import pathlib
import re
from typing import List, Dict
import subprocess
import IPython
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging


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
    def clean_srl(d: Dict, prefix: str = '') -> Dict:
        if len(d['verbs']) == 0:
            # print('unable to parse {}'.format(d['words']))
            return PatternUtils.get_roles_from_desc('', prefix=prefix)
        elif len(d['verbs']) == 1:
            return PatternUtils.get_roles_from_desc(d['verbs'][0]['description'], prefix=prefix)

        filtered_d = sorted(list(
            filter(lambda v: v['verb'] not in [],
                   filter(lambda v: 'description' in v.keys() and 'tags' in v.keys(),
                          d['verbs']
                          )
                   )
        ),
            key=lambda v: v['tags'].count('O')
        )
        return PatternUtils.get_roles_from_desc(filtered_d[0]['description'], prefix=prefix)

    @staticmethod
    def split_sentences(matches: List[str]):
        for m in matches:
            nltk.sent_tokenize(m)

    @staticmethod
    def check_pattern_in_file_grep(pattern: str, base_path: str, files_pattern: str) -> List[str]:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(**replacements)

        with open(pathlib.Path(os.getcwd()) / f'temp_bash.sh', 'w') as fp:
            fp.write(" ".join([
                'grep', '-iE', fr'"{regex_pattern}"',
                os.path.join(base_path, files_pattern),
                '-a', '> OUT'
                # '> temp_bash_output'
            ]))

        IPython.embed()
        exit()

        out = subprocess.run(['sh', 'temp_bash.sh'],
                             check=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)

        all_matches = []
        for line in out:
            m_list = re.findall(regex_pattern, line)
            for m in m_list:
                all_matches.append({
                    'line': line,
                    'pattern': pattern,
                    **{k: v for k, v in zip(pattern_keys, m)},
                })

    @staticmethod
    def check_pattern_in_files(pattern: str, base_path: str, files_pattern: str) -> List[str]:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(
            **replacements
        )

        all_matches = []
        for f in tqdm(pathlib.Path(base_path).iterdir(), desc='files'):
            if files_pattern is not None and files_pattern not in str(f):
                continue
            with open(f, 'r') as fp:
                for line in fp:
                    m_list = re.findall(regex_pattern, line)
                    for m in m_list:
                        all_matches.append({
                            'line': line,
                            'pattern': pattern,
                            **{k: v for k, v in zip(pattern_keys, m)},
                        })

        parser = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz",
            import_plugins=True
        )

        # IPython.embed()
        for match in tqdm(all_matches, desc='SRL'):
            for k in pattern_keys:
                match[f'parsed_{k}'] = PatternUtils.clean_srl(parser.predict(sentence=match[k]))

        # IPython.embed()
        return all_matches

    SINGLE_SENTENCE_DISABLING_PATTERNS = [
        r"^{action} unless {precondition}\.",
        r"\. {action} unless {precondition}\.",
        r"^{any_word} unless {precondition}, {action}\.",
        r"^{any_word} unless {precondition}, {action}\.",
        r"{negative_precondition} (?:so|hence|consequently) {action}\.",
    ]

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

    # PRECONDITION_REGEX = r'([\w\-\\\/\+\* ,\']+)'

    ENABLING_PATTERNS = [
        "{action} only if {precondition}.",
        "{precondition} (?:so|hence|consequently) {action}.",
        "{precondition} makes {action} possible.",
    ]

    DISABLING_WORDS = [
        "unless",
    ]




