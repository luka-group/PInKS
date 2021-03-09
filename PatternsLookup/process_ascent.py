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
    # with open(config.ascent_path) as fp:
    #     for line in tqdm(fp.readline(), desc='concepts'):
    #         IPython.embed()
    #         exit()
    #         concept = json.loads(line.replace("\'", "\""))
    #         for asrt in tqdm(_iter_assertions(concept), desc='assertions'):
    #             if config.predicate in asrt['predicate']:
    #                 output.append(asrt)

    # df_ascent = pd.read_json(config.ascent_path, lines=True,  chunksize=100)
    pbar_concept = tqdm(desc='concepts')
    pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True,  chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()
            for asrt in _iter_assertions(concept.to_dict()):
                if config.predicate in asrt['predicate']:
                    output.append(asrt)
                    pbar_assert.update()

    # for concept in tqdm(ascent, desc='concepts'):
    #     for asrt in tqdm(_iter_assertions(concept), desc='assertions'):
    #         if config.predicate in asrt['predicate']:
    #             output.append(asrt)

    IPython.embed()
    exit()
    logger.info(f'converting to pandas')
    df = pd.DataFrame(output)
    df.to_csv(config.output_name)
    return df


@hydra.main(config_path='../Configs/process_ascent_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # pd.read_json(config)
    extract_usedfor_assertions(config)


if __name__ == '__main__':
    main()
