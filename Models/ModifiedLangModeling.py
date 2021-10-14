import hydra

import omegaconf
import pytorch_lightning as pl
# from loguru import logger

# from LMBenchmarkEvaluator import BaseEvaluationModule
from DataModules.ModMLMData import ModMLMDataModule
from Modules.ModifiedLangModelingModule import ModifiedLMModule

import logging
logger = logging.getLogger(__name__)


# @hydra.main(config_path='../Configs/modified_lm_config.yaml')
@hydra.main(config_path="../Configs", config_name="modified_lm_config")
def main(config: omegaconf.dictconfig.DictConfig):

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


if __name__ == '__main__':
    main()
