#!/usr/bin/env bash

cd ..
export PYTHONPATH="$(pwd)"

python Models/Tune_Eval_NLI.py \
    model_setup.model_name="roberta-large-mnli" \
    model_setup.tuned_model_path="/nas/home/qasemi/CQplus/Outputs/ModifiedLangModeling/Checkpoint/ModifiedLMModule.ckpt" \
    +n_MNLI_samples='40000'\
    +nli_module_class='NLIModule' \
    +data_class="MnliTuneCqTestDataModule" \
    train_setup.do_train=true \
    hardware.gpus='1' \
    train_setup.batch_size=8 \
    hydra.run.dir="/nas/home/qasemi/CQplus/Outputs/Tune_Eval_NLI/\${nli_module_class}/\${data_class}"

#python NLI_Tune_Weak_Eval_CQ.py \
#    cq_path='/nas/home/qasemi/Mowgli-CoreQuisite/outputs/EvaluateBatch/MCQ-2000/BasicBenchmark/test.csv' \
#    model_setup.model_name="roberta-large-mnli" \
#    train_setup.do_train=true \
#    hardware.gpus='2' \
#    train_setup.batch_size=8
