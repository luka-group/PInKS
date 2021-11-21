# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:30:05 2021

@author: Dell
"""

import logging
import os
import pathlib
import pdb
import traceback
from typing import List

import IPython
import hydra
import nltk
import numpy as np
import omegaconf
import pandas as pd
from tqdm import tqdm

from SnorkelUtil import SnorkelUtil
from SnorkelOutputUtil import ProcessOutputUtil

nltk.download('punkt')
nltk.download("wordnet")

logger = logging.getLogger(__name__)


def ascent_extract_all_sentences_df(config: omegaconf.dictconfig.DictConfig):
    ascent_path = pathlib.Path(config.ascent_path).expanduser()
    logger.info(f'loading json from {ascent_path}')
    output = []
    all_sents = []

    pbar_concept = tqdm(desc='concepts')

    for df_chunk in pd.read_json(ascent_path, lines=True, chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()
            # create sources lut
            for k, s in concept['sentences'].items():
                all_sents.append(s['text'].replace('\n', ' '))

    logger.info(f'converting to pandas')
    df = pd.DataFrame(all_sents, columns=['text'])
    return df


@hydra.main(config_path="../Configs", config_name="process_dataset_using_snorkel")
def main(config: omegaconf.dictconfig.DictConfig):
    """With SnorkelUtil"""

    # df = _prepare_corpora(config).iloc[:500000]
    df = _prepare_corpora(config)

    snorkel_util = SnorkelUtil(config)
    snorkel_util.apply_labeling_functions(df)

    snorkel_util.add_action_precondition(df)

    df = df[df.label != SnorkelUtil.ABSTAIN]
    df.to_csv(config.output_names.snorkel_output, index=False)
    logger.info("Saved matches at:" + str(os.getcwd()) + str(config.output_names.snorkel_output))

    count = df["label"].value_counts()
    logger.info(f"\nLabel  Count\n{count}")

    logger.info('Save labeling matrix')
    np.save(str(pathlib.Path(config.output_names.labeling_matrix).expanduser()), snorkel_util.L)

    # Filtering
    ProcessOutputUtil.filter_dataset(config, df)

    IPython.embed()
    exit()
    # Data Augmentation
    logger.info('Augmenting data using BERT mask-filling.')
    ProcessOutputUtil.data_augmentation(config)

    # Extract Examples
    # logger.info("Saving Examples....")
    # examples_df = snorkel_util.return_examples(df)
    # examples_df.to_csv(config.output_examples, index=False)


def _prepare_omcs(config) -> List[str]:
    logger.info(f'Read OMCS data.')
    input_path = config.omcs_path
    df_omcs = pd.read_csv(input_path, sep="\t", error_bad_lines=False)
    logger.info(f'Read {len(df_omcs)} lines from OMCS')
    return df_omcs['text']


def _prepare_ascent(config) -> List[str]:
    logger.info(f'Prepare ASCENT data.')
    output_path = pathlib.Path(config.output_names.ascent_sentences_df).expanduser()
    if not output_path.exists():
        logger.info(f'Extracting ASCENT sentences from {config.ascent_path}')
        df_ascent = ascent_extract_all_sentences_df(config)
        df_ascent.to_csv(output_path, index=False)
    else:
        logger.info(f'Reading processed ASCENT sentences from: {output_path}')
        df_ascent = pd.read_csv(output_path)

    logger.info(f'Read {len(df_ascent)} lines from ASCENT')
    return df_ascent['text']


def _prepare_corpora(config) -> pd.DataFrame:
    text_rows = []
    logger.info("Processing Dataset: " + config.dataset_name)
    if "omcs" in config.dataset_name.lower():
        text_rows.extend(_prepare_omcs(config))

    if "ascent" in config.dataset_name.lower():
        text_rows.extend(_prepare_ascent(config))

    # df = pd.concat(df_list)
    logger.info("text_rows len=" + str(len(text_rows)))
    df = pd.DataFrame(text_rows, columns=['text'])

    df['text'] = df['text'].astype(str)

    logger.info(f"Merged Df head=\n{df.head()}")

    df.to_csv(config.output_names.extract_all_sentences_df, index=False)
    return df


if __name__ == '__main__':
    main()
