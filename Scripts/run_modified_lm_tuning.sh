#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"


python Models/ModifiedLangModeling.py \
    username=$(whoami) \
    lm_module.model_name_or_path="roberta-large" \
    data_module.train_file="/nas/home/qasemi/CQplus/Outputs/RemoveSimpleWeakPNLI/weakcq_filtered.csv"