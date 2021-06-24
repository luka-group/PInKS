#!/usr/bin/env bash

cd ../Models

python NLIEvaluator.py \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
    model_setup.model_name="roberta-large-mnli" \
    train_setup.do_train=true \
    hardware.gpus='2' \
    train_setup.batch_size=8
