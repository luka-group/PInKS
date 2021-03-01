import os

import IPython
import hydra
import omegaconf
import json

from Patterns import PatternUtils

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/lookup_pattern_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # files_patter = os.path.join([config.corpus_path, 'urlsf_subset*'])
    files_patern = '/urlsf_subset'

    matches = {}
    for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS:
        matches[pat] = PatternUtils.check_pattern_in_file(pat, config.corpus_path, files_patern)
    json.dump(matches, config.output_name)


if __name__ == '__main__':
    main()
