import pathlib

import IPython
import hydra
import omegaconf
import pandas as pd
import numpy as np

from transformers import AutoTokenizer, Pipeline, AutoModelForSequenceClassification

from tqdm import tqdm, tqdm_pandas
tqdm.pandas()

import logging
logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):

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

    weak_cq_path = pathlib.Path(config.weak_cq_path)
    logger.info('Loading data from {}'.format(weak_cq_path))
    df: pd.DataFrame = pd.read_csv(weak_cq_path).fillna('')
    df["text"] = df.apply(
        axis=1,
        func=lambda r: "{} </s></s> {}".format(r['action'], r['precondition'])
    )
    # id2label = {
    #     "0": "CONTRADICTION",
    #     "1": "NEUTRAL",
    #     "2": "ENTAILMENT"
    # },
    df['label'] = df['label'].apply(lambda l: {"CONTRADICT": 0, "ENTAILMENT": 2, 0: 0, 1: 2}[l])

    logger.info(f'compute nli results')
    df['nli_result'] = df['text'].progress_apply(lambda s: np.argmax(pipe(s)))

    df[df['nli_result'] != df['label']].to_csv(weak_cq_path.stem+'_filtered.csv')


if __name__ == '__main__':
    main()
