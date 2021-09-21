import json
import logging

import hydra
import omegaconf

from Patterns import PatternUtils

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/lookup_pattern_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # files_patern = os.path.join(config.corpus_path, 'urlsf_subset*')
    files_patern = 'urlsf_subset'
    matches = {}
    for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS:
        # matches[pat] = PatternUtils.check_pattern_in_files(pat, config.corpus_path, files_patern)
        matches[pat] = PatternUtils.check_pattern_in_file_grep(pat, config.corpus_path, files_patern)
    with open(config.output_name, 'w') as fp:
        json.dump(matches, fp)


if __name__ == '__main__':
    main()
