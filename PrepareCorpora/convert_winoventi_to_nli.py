from typing import Callable, List
from tqdm import tqdm
import IPython
import pandas as pd
import pathlib


out_dir = pathlib.Path('~/CQplus/Outputs/Other_NLI').expanduser()
df = pd.read_csv('../Outputs/Corpora/WINOVENTI/winoventi_bert_large_final.tsv', sep='\t')


# def flatmap(self, func: Callable[[pd.Series], pd.Series], ignore_index: bool=False):
#     return self.map(func).explode(ignore_index)


def wino_to_nli(wino: pd.Series) -> List[pd.Series]:
    sents = wino['masked_prompt'].replace('..', '.').split('. The ')
    assert len(sents) == 2, 'Odd pattern: {}'.format(sents)

    prec = f'The {sents[1]}'

    return [
        pd.Series(d) for d in
        [
            # normal bias
            {
                "premise": prec.replace('[MASK]', wino['target']),
                "hypothesis": sents[0],
                "label": 1
            },
            # Alternative
            {
                "premise": prec.replace('[MASK]', wino['incorrect']),
                "hypothesis": sents[0],
                "label": 0
            },

        ]
    ]


out_df_list = []
for i, r in tqdm(df.iterrows(), desc='wino examples'):
    try:
        out_df_list += wino_to_nli(r)
    except AssertionError as e:
        sent = r['masked_prompt']
        print(f'>>>> Skipping {sent}: {e}\n')

out_df = pd.DataFrame.from_dict(out_df_list)
out_df.to_csv(f'{out_dir}/winoventi_nli.csv', index=False)

