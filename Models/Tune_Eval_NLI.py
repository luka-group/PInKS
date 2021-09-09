import ast
import logging
from typing import List, Union, Dict

import IPython
import hydra
import omegaconf
import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune

from Models.DataModules.BaseNLIDataModule import WeakTuneCqTestDataModule, BaseNLIDataModule, MnliTuneCqTestDataModule
from Models.DataModules.CQOnlyNLIDataModule import CQOnlyNLIDataModule
from Models.Modules.BaseNLIModule import NLIModule
from Models.Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM
import Models.Utils as Utils

logger = logging.getLogger(__name__)


# def update_config(base: Dict, new: Dict):
#     if isinstance(base, dict):
#         for k, v in new.items():
#             if isinstance(v, dict):
#                 base[k] = update_config(base[k], v)
#             if k in base:
#                 base[k] = v
#         return {k: update_config(v, new) for k, v in base.items()}
#     else:
#         return base


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

    print("Best hyperparameters found were: ", analysis.best_config)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # _run_train_test(config)
    tune_nli_asha(config)


def _run_nli_train_test(config: omegaconf.dictconfig.DictConfig):
    # wandb.init()
    _model_class = {
        "NLIModuleWithTunedLM": NLIModuleWithTunedLM,
        "NLIModule": NLIModule,
    }[config.nli_module_class]
    _data_class_lut = {
        "BaseNLIDataModule": BaseNLIDataModule,
        "WeakTuneCqTestDataModule": WeakTuneCqTestDataModule,
        "MnliTuneCqTestDataModule": MnliTuneCqTestDataModule,
        "CQOnlyNLIDataModule": CQOnlyNLIDataModule,
    }
    data_class_name_list = [config.data_class] if isinstance(config.data_class, str) else list(config.data_class)
    assert isinstance(data_class_name_list, List), data_class_name_list
    _module = _model_class(config)
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
        callbacks=[
            pl.callbacks.EarlyStopping(monitor="val_loss", min_delta=1e-2),
            TuneReportCallback(
                {
                    "loss": "val_loss",
                    "mean_accuracy": "val_acc",
                    "f1": "val_f1",
                },
                on="validation_end")
        ],
    )
    for data_cls_name in data_class_name_list:
        _data_cls = _data_class_lut[data_cls_name]
        _data = _data_cls(config=config)
        logger.info(f'Tuning on {data_cls_name}')
        _module.extra_tag = 'fit'
        trainer.fit(_module, datamodule=_data)
    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    trainer.test(_module, datamodule=_data)


if __name__ == '__main__':
    main()
