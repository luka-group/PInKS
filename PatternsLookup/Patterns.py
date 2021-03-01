import pathlib
import re
from typing import List, Dict
import subprocess
import IPython
import nltk
from tqdm import tqdm
from allennlp.predictors.predictor import Predictor


class PatternUtils:
    @staticmethod
    def get_roles_from_desc(desc: str, prefix: str) -> Dict:
        if len(desc) == 0:
            args_dict = {}
        else:
            args_dict = dict(re.findall(r'\[([^:]+): ([^\]]+)]', desc))
        return {
            f'{prefix}_verbs': args_dict.get('V', ''),
            **{f'{prefix}_arg{i}': args_dict.get(f'ARG{i}', '') for i in range(1, 3)},
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
    def check_pattern_in_file(pattern: str, base_path: str, files_pattern: str) -> List[str]:
        pattern_keys = re.findall(r'\{([^\}]+)}', pattern)
        replacements = {k: PatternUtils.REPLACEMENT_REGEX[k] for k in pattern_keys}
        regex_pattern = pattern.format(
            **replacements
            # precondition=PatternUtils.PRECONDITION_REGEX,
            # action=PatternUtils.FACT_REGEX
        )

        # out = subprocess.run(['grep', '-i', fr'"{regex_pattern}"', files_pattern, '-a'],
        #                      check=True,
        #                      stdout=subprocess.PIPE,
        #                      stderr=subprocess.PIPE)
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
        IPython.embed()
        parser = Predictor.from_path(
            "https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")

        for match in tqdm(all_matches, desc='SRL'):
            for k in pattern_keys:
                match[f'parsed_{k}'] = PatternUtils.clean_srl(parser.predict(sentence=match[k]))

        IPython.embed()
        return all_matches

    SINGLE_SENTENCE_DISABLING_PATTERNS = [
        "{action} unless {precondition}.",
        "{negative_precondition} (?:so|hence|consequently) {action}.",
    ]

    REPLACEMENT_REGEX = {
        'action': r'([\w\-\\\/\+\* ,\']+)',
        'precondition': r'([\w\-\\\/\+\* ,\']+)',
        'negative_precondition': r'([\w\-\\\/\+\* ,\']+)',
    }
    FACT_REGEX = r'([\w\-\\\/\+\* ,\']+)'
    PRECONDITION_REGEX = r'([\w\-\\\/\+\* ,\']+)'

    ENABLING_PATTERNS = [
        "{action} only if {precondition}.",
        "{precondition} (?:so|hence|consequently) {action}.",
        "{precondition} makes {action} possible.",
    ]

    DISABLING_WORDS = [
        "unless",
    ]
