import itertools
from typing import Callable, List
from tqdm import tqdm
import IPython
import pandas as pd
import pathlib


out_dir = pathlib.Path('~/CQplus/Outputs/Other_NLI').expanduser()
train_path = pathlib.Path(
    '~/CQplus/Outputs/Corpora/ANION/anion_data/commonsense_contradict/commonsense_contradict_trn.csv'
).expanduser()


def replace_values_in_string(text, args_dict):
    for key in args_dict.keys():
        text = text.replace(key, str(args_dict[key]))
    return text


def anion_to_nli(ani: pd.Series) -> List[pd.Series]:
    person_lut = {
        'PersonX': 'Alice',
        'PersonY': 'Bob',
    }
    hypothesis_neg = replace_values_in_string(ani['event']+'.', person_lut)
    hypothesis_orig = replace_values_in_string(ani['original']+'.', person_lut)

    personx = person_lut['PersonX']
    x_effect = [f'{personx} {s}.' for s in ani['xEffect'] if s is not 'none']
    x_react = [f'{personx} is {s}.' for s in ani['xReact'] if s is not 'none']
    x_want = [f'{personx} wants {s}.' for s in ani['xWant'] if s is not 'none']
    x_intent = [f'{personx} wants {s}.' for s in ani['xIntent'] if s is not 'none']
    x_need = [f'{personx} needs {s}.' for s in ani['xIntent'] if s is not 'none']

    return [
        pd.Series({
            "premise": prem,
            "hypothesis": hyp,
            "label": lbl
        })
        for (hyp, lbl) in [(hypothesis_orig, 0), (hypothesis_neg, 1)]
        for prem in itertools.chain(x_effect, x_react, x_want, x_intent, x_need)
    ]


for split in ['train', 'test', 'dev']:
    file_path = str(train_path).replace('trn.csv', {
        'train': 'trn',
        'test': 'tst',
        'dev': 'dev',
    }[split] + '.csv')

    df = pd.read_csv(file_path).fillna('')
    out_df_list = []
    for i, r in tqdm(df.iterrows(), desc='wino examples', total=len(df)):
        try:
            out_df_list += anion_to_nli(r)
        except Exception as e:
            sent = r['event']
            print(f'>>>> Skipping {sent}: {e}\n')
            IPython.embed()
            exit()

    out_df = pd.DataFrame.from_dict(out_df_list)
    out_df.to_csv(f'{out_dir}/anion_nli_{split}.csv', index=False)

