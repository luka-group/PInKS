import hydra
import nltk
import omegaconf
import pandas as pd
from tqdm import tqdm

nltk.download("wordnet")

from SnorkelUtil import SnorkelUtil

import logging

logger = logging.getLogger(__name__)


class ProcessOutputUtil():

    @staticmethod
    def filter_dataset(merged_df):
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
        for index, row in tqdm(merged_df.iterrows()):
            if not (ProcessOutputUtil.isQuestion(row['text'])) and ProcessOutputUtil.hasVerb(
                    row['precondition']) and ProcessOutputUtil.isEnglish(row['text']):
                new_row = {"text": row['text'], "action": row['action'], "precondition": row['precondition'],
                           "label": row['label']}
                filtered_dataset = filtered_dataset.append(new_row, ignore_index=True)
                count += 1
        # print("Filtered True count="+str(count))
        print("Filtered len=" + str(len(filtered_dataset)))

        count = filtered_dataset["label"].value_counts()
        print("Label  Count")
        print(count)

        return filtered_dataset

    @staticmethod
    def isQuestion(text):

        text = text.strip()
    if ('?' in text) or (text.split()[0].lower() in question_start_words):
        return True
    return False

    @staticmethod
    def hasVerb(text):
        text = nltk.word_tokenize(text)
        result = nltk.pos_tag(text)
        for tags in result:
            if tags[1] in VERB_CODES:
                return True
        return False

    @staticmethod
    def isEnglish(text):
        if detect(text) == 'en':
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
    def merge_helper(df, text, actions, preconditions, labels):

        for index, row in tqdm(df.iterrows()):
            action = row["Action"]
            precondition = row["Precondition"]
            label = row["label"]
            if label == 2:
                continue
                # precondition=row['Precondition']
                # action=row['Action']
                # precondition, label = disambiguate(row["text"])
            text.append(row["text"])
            actions.append(action)
            preconditions.append(precondition)
            labels.append(label)
    return

    # @staticmethod
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


@hydra.main(config_path="../Configs", config_name="snorkel_output_util_config")
def main(config: omegaconf.dictconfig.DictConfig):
    if config.util_method == "filter":
        merged_df = pd.read_csv(config.merged_dataset)
        filtered_df = ProcessOutputUtil.filter_dataset(merged_df)
        filtered_dataset.to_csv(config.filtered_output_path)

    elif config.util_method == "merge":
        path1 = config.matches_path1
        path2 = config.matches_path2

        text = []
        actions = []
        preconditions = []
        labels = []

        df1 = pd.read_csv(config.matches_path1)
        df2 = pd.read_csv(config.matches_path2)

        ProcessOutputUtil.merge_helper(df1, text, actions, preconditions, labels)
        ProcessOutputUtil.merge_helper(df2, text, actions, preconditions, labels)

        final_merged_df = pd.DataFrame(list(zip(text, actions, preconditions, labels)),
                                       columns=['text', 'action', 'precondition', 'label'])

        final_merged_df.to_csv(config.merged_output_path)


if __name__ == '__main__':
    main()
