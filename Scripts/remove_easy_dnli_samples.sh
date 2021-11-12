#!/usr/bin/env bash

cd ../PrepareCorpora


python RemoveSimpleDNLI.py \
    model_setup.model_name="roberta-large-mnli" \
    dnli_path="/nas/home/qasemi/CQplus/Outputs/Other_NLI/dnli_nli_test.csv" \
    hardware.gpus='3'
