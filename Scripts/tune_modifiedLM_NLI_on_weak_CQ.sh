#!/usr/bin/env bash

cd ../Models

python NLI_LMEvaluator.py \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    train_setup.do_train=true \
    train_setup.batch_size=8
