#!/usr/bin/env bash

cd ../Models

python NLI_Tune_MNLI_Weak_Eval_CQ.py \
    model_setup.model_name="roberta-large-mnli" \
    +n_MNLI_samples='40000'\
    train_setup.do_train=true \
    hardware.gpus='1' \
    train_setup.batch_size=8


#