import os

import IPython
import hydra
import omegaconf
import json

from tqdm import tqdm

from Patterns import PatternUtils

import logging
logger = logging.getLogger(__name__)


# @hydra.main(config_path='../Configs/lookup_pattern_config.yaml')
@hydra.main(config_path="../Configs", config_name="lookup_pattern_config")
def main(config: omegaconf.dictconfig.DictConfig):
    # files_patern = os.path.join(config.corpus_path, 'urlsf_subset*')
    files_patern = "src"
    matches = {}
    for pat in PatternUtils.SINGLE_SENTENCE_DISABLING_PATTERNS:
        # matches[pat] = PatternUtils.check_pattern_in_files(pat, config.corpus_path, files_patern)
        matches[pat] = PatternUtils.check_pattern_in_files_gigaword(pat, config.corpus_path, files_patern)
    with open(config.output_name, 'w') as fp:
        json.dump(matches, fp)


if __name__ == '__main__':
    main()





# base_path=r"C:\Users\Dell\Desktop\Piyush\USC-CQ\Gigaword10k_msft\org_data\test_src"


# for f in tqdm(pathlib.Path(base_path).iterdir(), desc='files'):
#     if files_pattern is not None and files_pattern not in str(f):
#         continue
#     with open(f, 'r') as fp:
#         for line in tqdm(fp, desc='lines'):
#             print(line)
#             for sent in line:
#                 print(sent)
#                 # for ps in sent:
#                 #     print(ps)
#                 break
                
                
#             fix_d = {
#                 'pattern': pattern,
#                 'label': label
#             }
#             for new_match in PatternUtils.find_matches_in_line(line=line, regex_pattern=regex_pattern,
#                                                                pattern_keys=pattern_keys):
#                 all_matches.append({**new_match, **fix_d})


# for f in tqdm(pathlib.Path(base_path).iterdir(), desc='files'):
#     if files_pattern is not None and files_pattern not in str(f):
#         continue