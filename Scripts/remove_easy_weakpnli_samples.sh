#!/usr/bin/env bash

cd ../PrepareCorpora


python RemoveSimpleWeakPNLI.py \
    model_setup.model_name="roberta-large-mnli" \
    weak_cq_path="/nas/home/qasemi/CQplus/Outputs/CQWeak/final_v2_new_pat.csv" \
    hardware.gpus='3'
#    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_ascent/matched_sentences.csv' \
#    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
