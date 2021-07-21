#!/usr/bin/env bash

cd ../Models

python MaskedMLMNLI_Tune_Weak_Eval_CQ.py \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
    cq_path='/nas/home/qasemi/CQplus/Outputs/RemoveSimplePNLI/test_filtered.csv' \
    model_setup.model_name="roberta-large" \
    train_setup.do_train=true \
    hardware.gpus='2' \
    data_module.train_batch_size=2 \
    data_module.val_batch_size=2

