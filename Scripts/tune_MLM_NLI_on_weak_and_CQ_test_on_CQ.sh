#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +n_MNLI_samples='40000'\
    +nli_module_class='NLIModuleWithTunedLM' \
    +data_class=["WeakTuneCqTestDataModule","CQOnlyNLIDataModule"] \
    train_setup.do_train=true \
    hardware.gpus='2' \
    train_setup.batch_size=8 \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/\${nli_module_class}/WeakTuneCqTestDataModule_CQOnlyNLIDataModule"


#+data_class=["WeakTuneCqTestDataModule","CQOnlyNLIDataModule"] \