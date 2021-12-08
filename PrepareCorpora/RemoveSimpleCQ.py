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
    cq_main_path = pathlib.Path(config.cq_path).parent
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
        framework='pt'
    )
    pbar = tqdm(total=3)
    for subset in ['test', 'train', 'eval']:
        pbar.set_description(f'Filtering {subset}')
        cq_path = cq_main_path/f'{subset}.csv'
        logger.info('Loading data from {}'.format(cq_path))
        df: pd.DataFrame = pd.read_csv(cq_path).fillna('')
        df["text"] = df.apply(
            axis=1,
            func=lambda r: "{} </s></s> {}".format(r['question'], r['context'])
        )
        # id2label = {
        #     "0": "CONTRADICTION",
        #     "1": "NEUTRAL",
        #     "2": "ENTAILMENT"
        # },
        df['label'] = df['label'].apply(lambda l: {0: 0, 1: 2}[int(l)])

        logger.info(f'compute nli results')
        df['nli_result'] = df['text'].progress_apply(lambda s: np.argmax(pipe(s)))

        df[df['nli_result'] != df['label']].to_csv(cq_path.stem+'_filtered.csv')
        pbar.update()


if __name__ == '__main__':
    main()
