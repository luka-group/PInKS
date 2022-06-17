#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/4_2/*

#for TRAINDATA in weakcq
for TRAINDATA in atomic weakcq #cq winoventi anion atomic # dnli
do
  for TESTDATA in atomic
  do
    python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[${TRAINDATA}] \
        data_module.test_composition=[${TESTDATA}] \
        +weakcq_recal_threshold=0.90 \
        +n_${TESTDATA}_samples=50000 \
        +n_${TRAINDATA}_samples=50000 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="0" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=128 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/4_2"

  done
done
