#!/usr/bin/env bash

cd ../Models

python NLI_Tune_Weak_Eval_CQ.py \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
    cq_path='/nas/home/qasemi/CQplus/Outputs/RemoveSimplePNLI/test_filtered.csv' \
    mnli_path="/nas/home/qasemi/CQplus/Outputs/Corpora/MNLI/multinli_1.0/multinli_1.0_train.jsonl" \
    model_setup.model_name="roberta-large-mnli" \
    train_setup.do_train=true \
    hardware.gpus='2' \
    train_setup.batch_size=8


#