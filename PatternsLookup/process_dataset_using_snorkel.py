# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 12:30:05 2021

@author: Dell
"""

import logging
import os
import pathlib

import hydra
import nltk
import numpy as np
import omegaconf
import pandas as pd
from snorkel.labeling.model import LabelModel
from tqdm import tqdm

from SnorkelUtil import SnorkelUtil

nltk.download("wordnet")
logger = logging.getLogger(__name__)


def ascent_extract_all_sentences_df(config: omegaconf.dictconfig.DictConfig):
    # assert config.predicate == '*', f'{config.predicate}'
    logger.info(f'loading json from {config.ascent_path}')
    output = []
    all_sents = []

    pbar_concept = tqdm(desc='concepts')
    # pbar_assert = tqdm(desc=f'\"{config.predicate}\" assertions')

    for df_chunk in pd.read_json(config.ascent_path, lines=True, chunksize=100):
        for i, concept in df_chunk.iterrows():
            pbar_concept.update()

            # create sources lut
            # sent_dict = {}
            for k, s in concept['sentences'].items():
                all_sents.append(s['text'].replace('\n', ' '))

    logger.info(f'converting to pandas')
    print("Output Path:")
    print(config.ascent_output_names.extract_all_sentences_df)

    df = pd.DataFrame(all_sents, columns=['text'])
    df.to_csv(config.ascent_output_names.extract_all_sentences_df)
    # df.to_json('all_sentences.json')
    # df.to_json(config.ascent_output_names.extract_all_sentences, orient='records', lines=True)
    # return df


@hydra.main(config_path="../Configs", config_name="process_dataset_using_snorkel")
def main(config: omegaconf.dictconfig.DictConfig):
    """With SnorkelUtil"""

    input_path = ""
    df = pd.DataFrame()

    if config.dataset_name.lower() == "omcs":
        input_path = config.omcs_path
        df = pd.read_csv(input_path, sep="\t", error_bad_lines=False)
    elif config.dataset_name.lower() == "ascent":
        if config.ascent_method == 'extract_all_sentences_df':
            ascent_extract_all_sentences_df(config)
            return
        elif config.ascent_method == 'process_all_sentences_snorkel':
            # process_all_sentences_snorkel(config)
            input_path = pathlib.Path(os.getcwd()) / pathlib.Path(config.ascent_output_names.extract_all_sentences_df)
            print(input_path)
            df = pd.read_csv(input_path)

    # print(df.head())
    # for col in df.columns:
    #     print(col)
    # print("Text col Len="+str(len(df[',text'])))

    df['text'] = df['text'].astype(str)

    snorkel_util = SnorkelUtil(df)

    L, LFA_df = snorkel_util.get_L_matrix()

    label_model = LabelModel(cardinality=3, verbose=True)
    label_model.fit(L, n_epochs=config.snorkel_epochs, log_freq=50, seed=123)
    df["label"] = label_model.predict(L=L, tie_break_policy="abstain")

    df = snorkel_util.addActionPrecondition(L, LFA_df, df)
    df = df[df.label != SnorkelUtil.ABSTAIN]

    df.to_csv(config.output_name)

    count = df["label"].value_counts()
    print("Label  Count")
    print(count)

    with open('LabelingMatrix.npy', 'wb') as f:
        np.save(f, L)

    # Extract Examples
    # print("Saving Examples....")
    # examples_df=SnorkelUtil.returnExamples(L, LFA_df, df)
    # examples_df.to_csv(config.output_examples)    


if __name__ == '__main__':
    main()
