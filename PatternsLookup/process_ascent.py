import os
import pathlib
from typing import Dict, Generator

import pandas as pd
import IPython
import hydra
import omegaconf
import json
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def _iter_assertions(concept: Dict) -> Generator[Dict, None, None]:
    assert_keys = ['general_assertions', 'subgroup_assertions', 'aspect_assertions']
    for k in assert_keys:
        for asrt in concept[k]:
            for c in asrt['clusters']:
                for f in c['facets']:
                    yield {
                        'subject': c['subject'],
                        'predicate': c['predicate'],
                        'object': c['object'],
                        'facet_value': f["value"],
                        'facet_label': f["label"],
                    }


def extract_usedfor_assertions(config: omegaconf.dictconfig.DictConfig):
    logger.info(f'loading json from {config.ascent_path}')
    output = []

    pbar_concept = tqdm(desc='concepts')
    pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True,  chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()
            for asrt in _iter_assertions(concept.to_dict()):
                if config.predicate in asrt['predicate']:
                    output.append(asrt)
                    pbar_assert.update()

    logger.info(f'converting to pandas')
    df = pd.DataFrame(output)
    df.to_csv(config.output_names.extract_usedfor_assertions)
    return df


def extract_usedfor_sentences(config: omegaconf.dictconfig.DictConfig):
    logger.info(f'loading json from {config.ascent_path}')
    output = []

    pbar_concept = tqdm(desc='concepts')
    pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True, chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()

            # create sources lut
            sent_dict = {}
            for k, s in concept['sentences'].items():
                sent_dict[k] = s['text']

            # go through assertions
            assert_keys = ['general_assertions', 'subgroup_assertions', 'aspect_assertions']
            for k in assert_keys:
                for asrt in concept[k]:
                    for c in asrt['clusters']:
                        if config.predicate not in c['predicate']:
                            continue
                        if len(c['facets']) == 0:
                            continue

                        out_dict = {
                            'sources': [
                                sent_dict[sd['sentence_hash']]
                                for sd in c['sources']
                            ],
                            'facets': []
                        }
                        # out_dict['facets'] = []
                        for f in c['facets']:
                            out_dict['facets'].append({
                                'subject': c['subject'],
                                'predicate': c['predicate'],
                                'object': c['object'],
                                'facet_value': f["value"],
                                'facet_label': f["label"],
                            })


                        pbar_assert.update()
                        output.append(out_dict)

    logger.info(f'converting to pandas')
    df = pd.DataFrame(output)
    df.to_csv(config.output_names.extract_usedfor_sentences)
    return df


def extract_all_sentences(config: omegaconf.dictconfig.DictConfig):
    assert config.predicate == '*', f'{config.predicate}'
    logger.info(f'loading json from {config.ascent_path}')
    output = []

    pbar_concept = tqdm(desc='concepts')
    pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True, chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()

            # create sources lut
            sent_dict = {}
            for k, s in concept['sentences'].items():
                sent_dict[k] = s['text'].replace('\n', ' ')

            # go through assertions
            assert_keys = ['general_assertions', 'subgroup_assertions', 'aspect_assertions']
            for k in assert_keys:
                for asrt in concept[k]:
                    for c in asrt['clusters']:

                        out_dict = {
                            'subject': c['subject'].replace('\n', ' '),
                            'predicate': c['predicate'].replace('\n', ' '),
                            'object': c['object'].replace('\n', ' '),

                            'sources': [
                                sent_dict[sd['sentence_hash']]
                                for sd in c['sources']
                            ],
                            'facets': []
                        }
                        for f in c['facets']:
                            out_dict['facets'].append({
                                'label': f["label"].replace('\n', ' '),
                                'value': f["value"].replace('\n', ' '),
                            })

                        pbar_assert.update()
                        output.append(out_dict)

    logger.info(f'converting to pandas')
    df = pd.DataFrame(output)
    # df.to_json('all_sentences.json')
    df.to_json(config.output_names.extract_all_sentences, orient='records', lines=True)


def process_all_sentences(config: omegaconf.dictconfig.DictConfig):
    from Patterns import PatternUtils
    all_sents_path = pathlib.Path(os.getcwd())/pathlib.Path(config.output_names.extract_all_sentences)

    assert all_sents_path.exists(), all_sents_path

    matches = {}
    df_matches = []
    for p, label in [
        [r'{negative_precondition} {ENB_CONJ} {action}\.', 'CONTRADICT'],
        [r'\. {any_word} unless {precondition}, {action}\.', 'CONTRADICT'],
        [r'{precondition} makes {action} impossible.', 'CONTRADICT'],
        [r'{action} unless {precondition}\.', 'CONTRADICT'],
        # [r'{any_word} unless {precondition_action}\.', 'CONTRADICT'],

        [r'{action} only if {precondition}.', 'ENTAILMENT'],
        [r'{precondition} {ENB_CONJ} {action}.', 'ENTAILMENT'],
        [r'{precondition} makes {action} possible.', 'ENTAILMENT'],
    ]:
        matches[p] = PatternUtils.check_pattern_in_file_grep(
                p, base_path=os.getcwd(), files_pattern=config.output_names.extract_all_sentences,
                do_srl=config.process_all_sentences.do_srl, label=label)

        p_len = len(matches[p])
        logger.warning(f'Found {p_len} hits on {p}')

        df_matches.append(pd.DataFrame(matches[p]))

        logger.info(f'Dumping match results')
        with open(config.output_names.process_all_sentences, 'w') as fp:
            json.dump(matches, fp)

    logger.info("Dumping csv file")
    pd.concat(df_matches, axis=0).to_csv(config.output_names.process_all_sentences.replace('.json', '.csv'))

@hydra.main(config_path='../Configs/process_ascent_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    logger.warning(f'Config: {config}')

    if config.method == 'extract_all_sentences':
        extract_all_sentences(config)
    elif config.method == 'process_all_sentences':
        process_all_sentences(config)
    elif config.method == 'extract_usedfor_assertions':
        extract_usedfor_assertions(config)
    elif config.method == 'extract_usedfor_sentences':
        extract_usedfor_sentences(config)
    # pd.read_json(config)
    # extract_usedfor_assertions(config)
    # extract_usedfor_sentences(config)


if __name__ == '__main__':
    main()
