#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

#python Models/MLM_Tune_Weak_Eval_CQ.py \
#    model_setup.model_name="roberta-large-mnli" \
#    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
#    train_setup.do_train=true \
#    hardware.gpus='2' \
#    train_setup.batch_size=8
python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +n_MNLI_samples='40000'\
    +nli_module_class='NLIModuleWithTunedLM' \
    +data_class="WeakTuneCqTestDataModule" \
    train_setup.do_train=true \
    hardware.gpus='3' \
    train_setup.max_epochs=2 \
    train_setup.batch_size=8 \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/\${nli_module_class}/\${data_class}"
