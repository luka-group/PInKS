import hydra
import os, sys
from typing import List, Union, Dict
import omegaconf
import pytorch_lightning as pl
import ray
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

import Models.Utils as Utils
from Models.DataModules.ModMLMData import ModMLMDataModule
from Models.Modules.ModifiedLangModelingModule import ModifiedLMModule


import logging

logger = logging.getLogger(__name__)


def prepare_and_feed_config(updates: Dict, base_config: omegaconf.dictconfig.DictConfig):
    config = _get_updated_config(base_config, updates)
    config.pop('ray')

    # tune.report(loss=0.1)
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
        metric_columns=["loss"])

    analysis = tune.run(
        tune.with_parameters(
            prepare_and_feed_config,
            base_config=config
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
    logger.warning(f'Running _run_modlm_train_test')
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
    callbacks_list = [
        # pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4),
    ]
    if 'no_hyper_tune' not in config:
        callbacks_list.append(
            TuneReportCallback({
                "loss": "train_loss",
            }, on="batch_start")
        )

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


@hydra.main(config_path="../Configs", config_name="modified_lm_config")
def main(config: omegaconf.dictconfig.DictConfig):
    # tune_modlm_asha(config)
    # _run_modlm_train_test(config)


if __name__ == '__main__':
    main()
