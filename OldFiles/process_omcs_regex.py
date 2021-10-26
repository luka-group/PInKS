# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 22:58:07 2021

@author: Dell
"""

import json
import logging

import hydra
import omegaconf

from Patterns import PatternUtils

logger = logging.getLogger(__name__)


# @hydra.main(config_path='../Configs/lookup_pattern_config.yaml')
@hydra.main(config_path="../Configs", config_name="lookup_pattern_config")
def main(config: omegaconf.dictconfig.DictConfig):
    # files_patern = os.path.join(config.corpus_path, 'urlsf_subset*')
    files_patern = "omcs"
    matches = {}
    match_count = 0
    for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS:
        # matches[pat] = PatternUtils.check_pattern_in_files(pat, config.corpus_path, files_patern)
        matches[pat] = PatternUtils.check_pattern_in_files_omcs(pat, config.corpus_path, files_patern)
        match_count += len(matches[pat])
    with open(config.output_name, 'w') as fp:
        json.dump(matches, fp)

    print("Total Regex Matches:")
    print(match_count)


if __name__ == '__main__':
    main()
