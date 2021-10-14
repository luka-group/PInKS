import pathlib

import IPython
import hydra
import omegaconf
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, Pipeline, AutoModelForSequenceClassification
from tqdm import tqdm, tqdm_pandas

import logging
logger = logging.getLogger(__name__)
tqdm.pandas()


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):

    data_path = pathlib.Path(config.dnli_path)
    output_csv_name = 'dnli_filtered.csv'

    filter_data_and_dump(config, data_path, output_csv_name)


def filter_data_and_dump(config, data_path, output_csv_name):
    pipe = Pipeline(
        task="text-classification",
        model=AutoModelForSequenceClassification.from_pretrained(
            str(config.model_setup.model_name),
            cache_dir="/nas/home/qasemi/model_cache"
        ),
        tokenizer=AutoTokenizer.from_pretrained(
            str(config.model_setup.model_name),
            cache_dir="/nas/home/qasemi/model_cache",
            use_fast=False
        ),
        framework='pt',
        device=config.hardware.gpus,
    )
    weak_cq_path = data_path
    logger.info('Loading data from {}'.format(weak_cq_path))
    df: pd.DataFrame = pd.read_csv(weak_cq_path).fillna('')
    IPython.embed()
    df["text"] = df.apply(
        axis=1,
        func=lambda r: "{} </s></s> {}".format(r['question'], r['context'])
    )

    # id2label = {
    #     "0": "CONTRADICTION",
    #     "1": "NEUTRAL",
    #     "2": "ENTAILMENT"
    # },
    df['label'] = df['label'].apply(lambda l: {"CONTRADICT": 0, "ENTAILMENT": 2, 0: 0, 1: 2, 2: 2}[l])
    logger.info(f'compute nli results')
    df['nli_result'] = df['text'].progress_apply(lambda s: np.argmax(pipe(s)))
    df[df['nli_result'] != df['label']].to_csv(output_csv_name)


if __name__ == '__main__':
    main()
