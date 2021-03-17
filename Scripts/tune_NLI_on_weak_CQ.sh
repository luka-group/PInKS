#!/usr/bin/env bash

cd ../Models

python NLIEvaluator.py \
    benchmark_path='/nas/home/qasemi/CQplus/Outputs/process_ascent' \
    model_setup.model_name="roberta-large-mnli" \
    train_setup.do_train=true
