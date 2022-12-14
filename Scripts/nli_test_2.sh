#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    +nli_module_class='NLIModule' \
    data_module.train_composition=[weakcq] \
    data_module.test_composition=[atomic] \
    +weakcq_recal_threshold=0.95 \
    +n_weakcq_samples=50000 \
    +n_atomic_samples=50000 \
    data_module.use_class_weights=true \
    train_setup.do_train=true \
    hardware.gpus="3" \
    train_setup.max_epochs=5 \
    train_setup.batch_size=128 \
    +no_hyper_tune=true \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/Test_2"
