#!/usr/bin/env bash

rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/Test/*

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    +nli_module_class='NLIModule' \
    weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_dataset_using_snorkel/0.0/filtered_dataset_full.csv' \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    data_module.train_composition=[atomic] \
    data_module.test_composition=[atomic] \
    +weakcq_recal_threshold=0.90 \
    +n_weakcq_samples=50000 \
    data_module.overwrite_cache=true \
    data_module.use_class_weights=true \
    train_setup.do_train=true \
    hardware.gpus="1" \
    train_setup.max_epochs=5 \
    train_setup.batch_size=64 \
    train_setup.accumulate_grad_batches=2 \
    train_setup.learning_rate=1e-6 \
    +no_hyper_tune=true \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/Test"


#+no_hyper_tune=true \

#weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_dataset_using_snorkel/0.0/filtered_dataset_full.csv' \
#weak_cq_path='/nas/home/qasemi/CQplus/Outputs/process_dataset_using_snorkel/0.0/filtered_dataset_patterns_only.csv' \