#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/*

for TRAINDATA in winoventi anion weakcq dnli atomic
do
  for TESTDATA in dnli cq
  do
    python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[$TRAINDATA] \
        data_module.test_composition=[$TESTDATA] \
        +weakcq_recal_threshold=0.90 \
        +n_${TESTDATA}_samples=-1 \
        +n_${TRAINDATA}_samples=-1 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="1" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=32 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI"
  done
done
