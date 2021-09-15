#!/usr/bin/env bash

cd ../Models

python ModifiedLangModeling.py \
    lm_module.model_name_or_path="roberta-large" \
    data_module.train_file="/nas/home/pkhanna/CQplus/Outputs/filter_dataset/filtered_dataset.csv"