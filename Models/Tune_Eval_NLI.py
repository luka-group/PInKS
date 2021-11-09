import logging
from typing import List, Union, Dict

import IPython
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from Models.DataModules.NLIDataModule import NLIDataModule
from Models.Modules.BaseNLIModule import NLIModule
from Models.Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM
import Models.Utils as Utils

logger = logging.getLogger(__name__)


def prepare_and_feed_config(updates: Dict, base_config: omegaconf.dictconfig.DictConfig):
    config = _get_updated_config(base_config, updates)
    config.pop('ray')
    # logger.warning(f'config:\n{config}\n')
    # tune.report(loss=0.1, mean_accuracy=0.9, f1=0.2)
    # IPython.embed()
    # exit()
    _run_nli_train_test(config)


def _get_updated_config(base_config, updates):
    base_dict = Utils.flatten_config(base_config)
    base_dict.update(updates)
    base_config_dict = Utils.unflatten_config(base_dict)

    config = omegaconf.dictconfig.DictConfig(content=base_config_dict)
    # d = dict(base_config).update(updates)
    # config = Namespace(**update_config(dict(base_config), update_dict))
    return config


def tune_nli_asha(config: omegaconf.dictconfig.DictConfig):
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
        max_t=config.train_setup.max_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = tune.CLIReporter(
        # parameter_columns=["learning_rate", "train_batch_size", "max_seq_length"],
        metric_columns=["loss", "mean_accuracy", "f1"])

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
        name="tune_nli_asha")

    logger.warning("Best hyperparameters found were: ", analysis.best_config)


def _run_nli_train_test(config: omegaconf.dictconfig.DictConfig):
    # wandb.init()
    _module = {
        "NLIModuleWithTunedLM": NLIModuleWithTunedLM,
        "NLIModule": NLIModule,
    }[config.nli_module_class](config)

    # _module = _model_class(config)

    assert 'train_composition' in config.data_module, f'Invalid: {config.data_module}'
    _data = NLIDataModule(config=config)

    callbacks_list = [
        pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-4),

    ]

    if 'no_hyper_tune' not in config:
        callbacks_list.append(
            TuneReportCallback({
                "loss": "val_loss",
                "mean_accuracy": "val_acc",
                "f1": "val_f1",
            }, on="validation_end")
        )

    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=str(config['hardware']['gpus']),
        # gpus="2,3",
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['train_setup']['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config['train_setup']['warmup_steps'],
        resume_from_checkpoint=None,
        distributed_backend=None,
        logger=pl.loggers.WandbLogger(name=f"CQplus_NLI", project="CQ++"),
        callbacks=callbacks_list,
    )

    logger.info(f'Tuning on {config.data_module.train_composition}')
    _module.extra_tag = 'fit'
    trainer.fit(_module, datamodule=_data)

    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    logger.info(f'Testing on {config.data_module.test_composition}')
    trainer.test(_module, datamodule=_data)



@hydra.main(config_path="../Configs", config_name="model_evaluator_config")
def main(config: omegaconf.dictconfig.DictConfig):
    # _run_train_test(config)
    if 'ray' in config and 'no_hyper_tune' not in config:
        tune_nli_asha(config)
    else:
        _run_nli_train_test(config)


if __name__ == '__main__':
    main()
