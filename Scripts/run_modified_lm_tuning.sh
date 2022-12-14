#!/usr/bin/env bash


cd ..
export PYTHONPATH="$(pwd)"


python Models/ModifiedLangModeling.py \
    username=$(whoami) \
    lm_module.model_name_or_path="roberta-large" \
    data_module.train_file="/nas/home/$(whoami)/CQplus/Outputs/process_dataset_using_snorkel/0.0/precoditions_corpus.csv"
#    data_module.train_file="/nas/home/$(whoami)/CQplus/Outputs/process_dataset_using_snorkel/0.7/filtered_dataset.csv"