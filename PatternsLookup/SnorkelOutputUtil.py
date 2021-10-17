import IPython
import hydra
import nltk
import omegaconf
import pandas as pd
from tqdm import tqdm
import os

import json

from SnorkelUtil import SnorkelUtil
import langdetect

import logging

from transformers import pipeline
import random

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nltk.download("wordnet")
logger = logging.getLogger(__name__)


class ProcessOutputUtil:
    unmasker = pipeline('fill-mask', model='bert-base-uncased')

    @staticmethod
    def filter_dataset(config: omegaconf.dictconfig.DictConfig, df: pd.DataFrame):
        df_valid = df.dropna(axis=0)
        # df = pd.read_csv(config.merged_dataset)

        question_start_words = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]
        VERB_CODES = {
            'VB',  # Verb, base form
            'VBD',  # Verb, past tense
            'VBG',  # Verb, gerund or present participle
            'VBN',  # Verb, past participle
            'VBP',  # Verb, non-3rd person singular present
            'VBZ',  # Verb, 3rd person singular present
        }

        column_names = ["text", "action", "precondition", "label"]
        filtered_dataset = pd.DataFrame(columns=column_names)

        count = 0
        for index, row in tqdm(df_valid.iterrows(), desc='Filter Dataset'):
            if not (ProcessOutputUtil.isQuestion(row['text'])) and ProcessOutputUtil.hasVerb(
                    row['precondition']) and ProcessOutputUtil.isEnglish(row['text']):
                new_row = {"text": row['text'], "action": row['action'], "precondition": row['precondition'],
                           "label": row['label']}
                filtered_dataset = filtered_dataset.append(new_row, ignore_index=True)
                count += 1
        # print("Filtered True count="+str(count))
        logger.info("Filtered len=" + str(len(filtered_dataset)))

        count = filtered_dataset["label"].value_counts()
        logger.info(f"\nLabel  Count\n{count}")

        # Removing duplicate rows
        filtered_dataset = filtered_dataset.drop_duplicates()

        filtered_dataset.to_csv(config.output_names.filtered_output_path, index=False)


    @staticmethod
    def isQuestion(text):
        question_start_words = ["who", "what", "when", "where", "why", "how", "is", "can", "does", "do"]
        text = text.strip()
        if ('?' in text) or (text.split()[0].lower() in question_start_words):
            return True
        return False

    @staticmethod
    def hasVerb(text):
        try:
            text = nltk.word_tokenize(text)
        except Exception as e:
            logger.error(e)
            IPython.embed()
            exit()

        VERB_CODES = {
            'VB',  # Verb, base form
            'VBD',  # Verb, past tense
            'VBG',  # Verb, gerund or present participle
            'VBN',  # Verb, past participle
            'VBP',  # Verb, non-3rd person singular present
            'VBZ',  # Verb, 3rd person singular present
        }
        result = nltk.pos_tag(text)
        for tags in result:
            if tags[1] in VERB_CODES:
                return True
        return False

    @staticmethod
    def isEnglish(text):
        if langdetect.detect(text) == 'en':
            return True
        return False

    @staticmethod
    def containsIf(text):
        if_not_pat = "{action} if not {precondition}"
        if SnorkelUtil.pattern_exists(if_not_pat, text):
            return False
        elif SnorkelUtil.pattern_exists("{action} if {precondition}", text):
            return True
        return False

    @staticmethod
    def containsBut(text):
        pat = "{action} but {precondition}"
        if SnorkelUtil.pattern_exists(pat, text):
            return True
        return False

    @staticmethod
    def data_augmentation(config: omegaconf.dictconfig.DictConfig):
        
        filtered_df=pd.read_csv(config.output_names.filtered_output_path)
        aug_sents = []
        count = 0
        # logger.info("Current working dir="+os.getcwd())
        logger.info("Filtered DF len="+str(len(filtered_df)))
        for index, row in tqdm(filtered_df.iterrows()):
            try:
                aug_sents.extend(ProcessOutputUtil.fill_mask(row['text'], row['label']))
                count += 1
            except Exception as e:
                print(e)
                continue

            if index % 20000 == 0:
                with open(config.output_names.augmented_dataset_path, "w") as outfile:
                    json.dump(aug_sents, outfile)
                print("Saved at index=" + str(index))

        # Saving for the final time
        with open(config.output_names.augmented_dataset_path, "w") as outfile:
            json.dump(aug_sents, outfile)

        logger.info("Count of augmented sentence=" + str(count))

    
    @staticmethod
    def fill_mask(text, label):
        NOUN_CODES = {'NN', 'NNS', 'NNPS', 'NNP'}
        ADJECTIVE_CODES = {"JJ", "JJR", "JJS"}
        LEAVE_OUT_WORDS = {'understand', 'event', 'important', 'know', 'statement', 'true'}

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
                unmasked_list = list(ProcessOutputUtil.unmasker(new_sent_masked))[:3]
                new_aug_dict['predictions'] = unmasked_list
                aug_sents_dicts.append(new_aug_dict)

        aug_sents_dicts = ProcessOutputUtil.aug_selection_strategy(aug_sents_dicts, 15)
        return aug_sents_dicts

    @staticmethod
    def aug_selection_strategy(aug_sents_dicts, n_samples):
        return random.sample(aug_sents_dicts, min(n_samples, len(aug_sents_dicts)))





    # @staticmethod
    # def merge(config: omegaconf.dictconfig.DictConfig):
    #     path1 = config.filtered_omcs_path
    #     path2 = config.filtered_ascent_path
    #
    #     text = []
    #     actions = []
    #     preconditions = []
    #     labels = []
    #
    #     df1 = pd.read_csv(path1)
    #     df2 = pd.read_csv(path2)
    #
    #     ProcessOutputUtil.merge_helper(df1, text, actions, preconditions, labels)
    #     ProcessOutputUtil.merge_helper(df2, text, actions, preconditions, labels)
    #
    #     final_merged_df = pd.DataFrame(list(zip(text, actions, preconditions, labels)),
    #                                    columns=['text', 'action', 'precondition', 'label'])
    #
    #     final_merged_df.to_csv(config.merged_output_path, axis=False)

    # @staticmethod
    # def merge_helper(df, text, actions, preconditions, labels):
    #     for index, row in tqdm(df.iterrows()):
    #         action = row["action"]
    #         precondition = row["precondition"]
    #         label = row["label"]
    #         if label == 2:
    #             continue
    #             # precondition=row['Precondition']
    #             # action=row['Action']
    #             # precondition, label = disambiguate(row["text"])
    #         text.append(row["text"])
    #         actions.append(action)
    #         preconditions.append(precondition)
    #         labels.append(label)
    #     return
    #
    # def create_lf_annotation(filtered_df):
    #     for index,row in tqdm(filtered_df.iterrows()):
    #     text=row["text"]
    #     found=False
    #     for key,val in disabling_dict.items():
    #         if pattern_exists(val,text):
    #             lf_instances[key].append(text)
    #             found=True
    #             break
    #     if found:
    #         continue
    #     for key,val in enabling_dict.items():
    #         if pattern_exists(val,text):
    #             lf_instances[key].append(text)
    #             found=True
    #             break

    #     lf_instances_df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in lf_instances.items() ]))

    #     for key,val in lf_instances.items():
    #         label_row_name=key+" label"
    #         lf_instances_df[label_row_name]=""

    #     lf_instances_df = lf_instances_df.reindex(sorted(lf_instances_df.columns), axis=1)

    #     lf_instances_df.head(100).to_csv(output_path)



