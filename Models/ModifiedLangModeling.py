import hydra
import os, sys
from typing import List, Union, Dict
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

# from loguru import logger
import Models.Utils as Utils
from DataModules.ModMLMData import ModMLMDataModule
from Modules.ModifiedLangModelingModule import ModifiedLMModule
# from LMBenchmarkEvaluator import BaseEvaluationModule


import logging
logger = logging.getLogger(__name__)


def prepare_and_feed_config(updates: Dict, base_config: omegaconf.dictconfig.DictConfig):
    config = _get_updated_config(base_config, updates)
    config.pop('ray')

    # logger.warning(f'config:\n{config}\n')
    # tune.report(loss=0.1, mean_accuracy=0.9, f1=0.2)
    # IPython.embed()
    # exit()

    _run_modlm_train_test(config)


def _get_updated_config(base_config, updates):
    base_dict = Utils.flatten_config(base_config)
    base_dict.update(updates)
    base_config_dict = Utils.unflatten_config(base_dict)

    config = omegaconf.dictconfig.DictConfig(content=base_config_dict)
    # d = dict(base_config).update(updates)
    # config = Namespace(**update_config(dict(base_config), update_dict))
    return config


def tune_modlm_asha(config: omegaconf.dictconfig.DictConfig):
    assert "ray" in config, f'config does not have `ray`: {config}'
    # sweep_dict = {
    #     "train_setup.learning_rate": tune.loguniform(1e-6, 1e-4),
    #     # "data_module": {
    #     #     'train_batch_size': tune.choice([2, 4]),
    #     #     'max_seq_length': tune.choice([200, 300, 400, 450]),
    #     # },
    # }
    # setup the sweep values based on config
    sweep_dict = {}
    for k, v in dict(config.ray.sweep_dict).items():
        exec(f'sweep_dict[k] = {v}')

    scheduler = tune.schedulers.ASHAScheduler(
        max_t=1,
        grace_period=1,
        reduction_factor=2)

    reporter = tune.CLIReporter(
        # parameter_columns=["learning_rate", "train_batch_size", "max_seq_length"],
        metric_columns=["loss", "mean_accuracy", "f1"])

    analysis = tune.run(
        tune.with_parameters(
            _run_modlm_train_test
            # prepare_and_feed_config,
            # base_config=config
            ),
        resources_per_trial={
            "cpu": 1,
            "gpu": 4
        },
        metric="loss",
        mode="min",
        config=sweep_dict,
        num_samples=config.ray.num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_modlm_asha")

    logger.warning("Best hyperparameters found were: ", analysis.best_config)



def _run_modlm_train_test(config: omegaconf.dictconfig.DictConfig):


    # print('TUNE_EX\n\n', sys.path, '\n\n')

    # ------------
    # data
    # ------------
    data_module = ModMLMDataModule(config)

    # ------------
    # model
    # ------------
    lmmodel = ModifiedLMModule(config)

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        auto_select_gpus=True,
        max_epochs=1,
        min_epochs=1,
        resume_from_checkpoint=None,
        distributed_backend=None,
        accumulate_grad_batches=config['trainer_args']['accumulate_grad_batches'],
        limit_train_batches=config['trainer_args']['limit_train_batches'],
        logger=pl.loggers.WandbLogger(name=f"Mod-MLM", project="CQ++"),
    )

    trainer.fit(lmmodel, datamodule=data_module)
    trainer.save_checkpoint('Checkpoint/ModifiedLMModule.ckpt')
    trainer.test(lmmodel, datamodule=data_module)



# @hydra.main(config_path='../Configs/modified_lm_config.yaml')
@hydra.main(config_path="../Configs", config_name="modified_lm_config")
def main(config: omegaconf.dictconfig.DictConfig):
    tune_modlm_asha(config)
    # _run_modlm_train_test(config)



if __name__ == '__main__':
    main()
