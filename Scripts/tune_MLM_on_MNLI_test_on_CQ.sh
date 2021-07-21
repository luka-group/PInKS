#!/usr/bin/env bash

cd ../Models

python MLM_Tune_MNLI_Eval_CQ.py \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
    mnli_path="/nas/home/qasemi/CQplus/Outputs/Corpora/MNLI/multinli_1.0/multinli_1.0_train.jsonl" \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    train_setup.do_train=true \
    train_setup.batch_size=8
