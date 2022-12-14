#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/$(whoami)/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +nli_module_class='NLIModuleWithTunedLM' \
    data_module.train_composition=[mnli,weakcq] \
    data_module.test_composition=[cq] \
    +weakcq_recal_threshold=0.90 \
    +n_weakcq_samples=100000 \
    +n_anion_samples=100000 \
    +n_winoventi_samples=100000 \
    +n_atomic_samples=100000 \
    +n_mnli_samples=100000 \
    +n_cq_samples=500 \
    data_module.use_class_weights=true \
    train_setup.do_train=true \
    hardware.gpus='1' \
    train_setup.batch_size=32 \
    hydra.run.dir="/nas/home/$(whoami)/CQplus/Outputs/Tune_Eval_NLI/\${nli_module_class}/BaseNLIDataModule_CQOnlyNLIDataModule"


#+data_class=["WeakTuneCqTestDataModule","CQOnlyNLIDataModule"] \