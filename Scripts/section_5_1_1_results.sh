#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_1_1/*

for TRAINDATA in weakcq
do
  for TESTDATA in dnli cq winoventi anion atomic
  do
    python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[$TRAINDATA,$TESTDATA] \
        data_module.train_strategy='curriculum' \
        data_module.test_composition=[$TESTDATA] \
        +weakcq_recal_threshold=0.90 \
        +n_${TESTDATA}_samples=50000 \
        +n_${TRAINDATA}_samples=50000 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="0" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=64 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_1_1"
  done
done
