#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"


for STRATEGY in 'curriculum' ''
do
    rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_3/*

    python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[weakcq] \
        data_module.train_strategy="${STRATEGY}" \
        data_module.test_composition=[cq] \
        +weakcq_recal_threshold=0.9 \
        +n_weakcq_samples=50000 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="3" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=64 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_4"
done
