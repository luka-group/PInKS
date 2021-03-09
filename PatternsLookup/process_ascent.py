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
    df.to_csv(config.output_name)
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
            # for asrt in concept['sentences']:
            #     if config.predicate in asrt['predicate']:
            #         output.append(asrt)
            #         pbar_assert.update()

    IPython.embed()
    exit()

    logger.info(f'converting to pandas')
    df = pd.DataFrame(output)
    df.to_csv(config.output_name)
    return df


@hydra.main(config_path='../Configs/process_ascent_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # pd.read_json(config)
    # extract_usedfor_assertions(config)
    extract_usedfor_sentences(config)


if __name__ == '__main__':
    main()
