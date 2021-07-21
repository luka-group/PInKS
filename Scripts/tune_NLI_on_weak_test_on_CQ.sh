#!/usr/bin/env bash

cd ../Models

python NLI_Tune_Weak_Eval_CQ.py \
    model_setup.model_name="roberta-large-mnli" \
    train_setup.do_train=true \
    hardware.gpus='2' \
    train_setup.batch_size=8


#