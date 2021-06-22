#!/usr/bin/env bash

cd ../Models

python ModifiedLangModeling.py \
    lm_module.model_name_or_path="roberta-base" \
    data_module.train_file="/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_textonly_sentences.csv"