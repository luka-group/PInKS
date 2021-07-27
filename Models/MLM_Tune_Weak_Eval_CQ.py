import logging

import IPython
import hydra
import omegaconf
import pytorch_lightning as pl

from DataModules.BaseNLIDataModule import WeakTuneCqTestDataModule
from Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM
import Models.Utils as Utils

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModuleWithTunedLM(config)
    _data = WeakTuneCqTestDataModule(config)
    Utils.PLModelDataTest(model=_module, data=_data).run()
    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=str(config['hardware']['gpus']),
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['train_setup']['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config['train_setup']['warmup_steps'],
        resume_from_checkpoint=None,
        distributed_backend=None,
    )

    logger.info('Tuning')
    _module.extra_tag = 'fit'
    trainer.fit(_module, datamodule=_data)

    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    trainer.test(_module, datamodule=_data)


if __name__ == '__main__':
    main()
