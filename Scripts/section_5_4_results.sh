#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

#for TRAINDATA in weakcq
#do
#  for TESTDATA in dnli cq winoventi anion atomic
#  do
#    rm -rf /nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_4/*
#    python Models/Tune_Eval_NLI.py \
#        model_setup.model_name="roberta-large-mnli" \
#        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
#        +nli_module_class='NLIModule' \
#        data_module.train_composition=[$TRAINDATA,$TESTDATA] \
#        data_module.train_strategy='curriculum' \
#        data_module.test_composition=[$TESTDATA] \
#        +weakcq_recal_threshold=0.90 \
#        +n_${TESTDATA}_samples=50000 \
#        +n_${TRAINDATA}_samples=50000 \
#        data_module.use_class_weights=true \
#        train_setup.do_train=true \
#        hardware.gpus="3" \
#        train_setup.max_epochs=5 \
#        train_setup.batch_size=64 \
#        +no_hyper_tune=true \
#        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_4"
#  done
#done

python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[weakcq,cq] \
        data_module.train_strategy='curriculum' \
        data_module.test_composition=[cq] \
        +weakcq_recal_threshold=0.90 \
        +n_cq_samples=50000 \
        +n_weakcq_samples=50000 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="3" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=64 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_4"


python Models/Tune_Eval_NLI.py \
        model_setup.model_name="roberta-large-mnli" \
        model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
        +nli_module_class='NLIModule' \
        data_module.train_composition=[weakcq,cq] \
        data_module.train_strategy='multitask' \
        data_module.test_composition=[cq] \
        +weakcq_recal_threshold=0.90 \
        +n_cq_samples=50000 \
        +n_weakcq_samples=50000 \
        data_module.use_class_weights=true \
        train_setup.do_train=true \
        hardware.gpus="3" \
        train_setup.max_epochs=5 \
        train_setup.batch_size=64 \
        +no_hyper_tune=true \
        hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/5_4"
