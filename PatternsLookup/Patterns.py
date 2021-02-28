import pathlib
import re
from typing import List
import subprocess
import IPython
import nltk


class PatternUtils:
    @staticmethod
    def split_sentences(self, matches: List[str]):
        for m in matches:
            nltk.sent_tokenize(m)

    @staticmethod
    def check_pattern_in_file(pattern: str, base_path: str, files_pattern: str) -> List[str]:

        regex_pattern = pattern.format(
            precondition=PatternUtils.PRECONDITION_REGEX,
            action=PatternUtils.FACT_REGEX
        )

        IPython.embed()
        exit()
        # out = subprocess.run(['grep', '-i', fr'"{regex_pattern}"', files_pattern, '-a'],
        #                      check=True,
        #                      stdout=subprocess.PIPE,
        #                      stderr=subprocess.PIPE)
        for f in pathlib.Path(base_path).iterdir():
            if files_pattern not in f:
                continue
            with open(f, 'r') as fp:
                for line in fp:
                    m = re.match(regex_pattern, line)


        return str(out).split('\n')

    SINGLE_SENTENCE_DISABLING_PATTERNS = [
        "{action} unless {precondition}.",
        "{negative_precondition} (?:so|hence|consequently) {action}.",
    ]

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
