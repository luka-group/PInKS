#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +nli_module_class='NLIModule' \
    data_module.train_composition=[weakcq,dnli] \
    data_module.train_strategy='curriculum' \
    data_module.test_composition=[dnli] \
    +weakcq_recal_threshold=0.90 \
    +n_weakcq_samples=100000 \
    +n_dnli_samples=-1 \
    +n_cq_samples=500 \
    data_module.use_class_weights=true \
    train_setup.do_train=true \
    hardware.gpus="1" \
    train_setup.max_epochs=5 \
    train_setup.batch_size=32 \
    +no_hyper_tune=true \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI"
