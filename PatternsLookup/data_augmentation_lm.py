# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 00:48:13 2021

@author: Dell
"""

import json

import nltk
import pandas as pd
from tqdm import tqdm

nltk.download('punkt')
from transformers import pipeline
import random

nltk.download("wordnet")
nltk.download('averaged_perceptron_tagger')

unmasker = pipeline('fill-mask', model='bert-base-uncased')

NOUN_CODES = {'NN', 'NNS', 'NNPS', 'NNP'}
ADJECTIVE_CODES = {"JJ", "JJR", "JJS"}

LEAVE_OUT_WORDS = {'understand', 'event', 'important', 'know', 'statement', 'true'}


def aug_selection_strategy(aug_sents_dicts, n_samples):
    return random.sample(aug_sents_dicts, min(n_samples, len(aug_sents_dicts)))


def fill_mask(text, label):
    aug_sents_dicts = []
    text = nltk.word_tokenize(text)
    result = nltk.pos_tag(text)
    for idx, (word, tag) in enumerate(result):
        if word.lower() in LEAVE_OUT_WORDS:
            continue
        tmp_result = [list(ele) for ele in result]
        if (tag in NOUN_CODES) or (tag in ADJECTIVE_CODES):
            new_aug_dict = {
                'original_sentence': ' '.join(word[0] for word in tmp_result),
                'masked_word': tmp_result[idx][0],
                'masked_position': idx,
                'label': label
            }
            tmp_result[idx][0] = "[MASK]"
            new_sent_masked = ' '.join(word[0] for word in tmp_result)
            unmasked_list = list(unmasker(new_sent_masked))[:3]
            new_aug_dict['predictions'] = unmasked_list
            aug_sents_dicts.append(new_aug_dict)

    aug_sents_dicts = aug_selection_strategy(aug_sents_dicts, 15)
    return aug_sents_dicts


def main():
    augmented_dataset_path = "/nas/home/pkhanna/CQplus/Outputs/data_augmentation_lm/augmented_dataset_v2.json"
    filtered_dataset_path = "/nas/home/pkhanna/CQplus/Outputs/filter_dataset/filtered_dataset.csv"
    visited_path = "/nas/home/pkhanna/CQplus/Outputs/data_augmentation_lm/visited_v2.json"

    # print("Output Path= "+config.augmented_dataset_path)
    # print("Input Path="+config.augmented_dataset_input_path)

    try:
        with open(augmented_dataset_path) as f:
            aug_sents = json.load(f)

        with open(visited_path) as f:
            visited_dict = json.load(f)

        print("Found pre-saved file!")
    except:
        print("Searched file at=" + str(augmented_dataset_path))
        aug_sents = []
        visited_dict = {}

    df = pd.read_csv(filtered_dataset_path)
    count = 0

    for index, row in tqdm(df.iterrows()):
        if row['text'] not in visited_dict:
            try:
                aug_sents.extend(fill_mask(row['text'], row['label']))
                count += 1
                visited_dict[row['text']] = True
            except Exception as e:
                print(e)
                continue

        if index % 20000 == 0:
            with open(augmented_dataset_path, "w") as outfile:
                json.dump(aug_sents, outfile)

            with open(visited_path, "w") as outfile:
                json.dump(visited_dict, outfile)
            print("Saved at index=" + str(index))

    # Saving for the final time
    with open(augmented_dataset_path, "w") as outfile:
        json.dump(aug_sents, outfile)

    with open(visited_path, "w") as outfile:
        json.dump(visited_dict, outfile)

    print("Sentence augmented=" + str(count))


if __name__ == '__main__':
    main()
