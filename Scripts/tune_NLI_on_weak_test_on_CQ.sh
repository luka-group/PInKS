#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +nli_module_class='NLIModule' \
    data_module.train_composition=[weakcq] \
    +n_weakcq_samples=5000 \
    +n_dnli_samples=5000 \
    +n_atomic_samples=5000 \
    +n_mnli_samples=5000 \
    data_module.use_class_weights=true \
    train_setup.do_train=true \
    hardware.gpus="2" \
    train_setup.max_epochs=1 \
    train_setup.batch_size=4 \
    +no_hyper_tune=ture \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI"
#    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/\${nli_module_class}/\${data_class}"

#    data_module.train_composition=[dnli,atomic,mnli,weakcq] \

#python NLI_Tune_Weak_Eval_CQ.py \
#    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
#    model_setup.model_name="roberta-large-mnli" \
#    train_setup.do_train=true \
#    hardware.gpus='2' \
#    train_setup.batch_size=8
