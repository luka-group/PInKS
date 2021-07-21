#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/MLM_Tune_MNLI_Eval_CQ.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    train_setup.do_train=true \
    hardware.gpus='3' \
    train_setup.batch_size=8

#cq_path='/nas/home/qasemi/CQplus/Outputs/RemoveSimplePNLI/test_filtered.csv' \