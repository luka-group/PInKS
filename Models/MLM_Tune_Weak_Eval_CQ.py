import logging
import hydra
import omegaconf
import pytorch_lightning as pl

from DataModules.BaseNLIDataModule import WeakTuneCqTestDataModule
from Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModuleWithTunedLM(config)
    _data = WeakTuneCqTestDataModule(config)

    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        accumulate_grad_batches=config['train_setup']["accumulate_grad_batches"],
        max_epochs=config['train_setup']["max_epochs"],
        min_epochs=1,
        val_check_interval=config['train_setup']['val_check_interval'],
        weights_summary='top',
        num_sanity_val_steps=config['train_setup']['warmup_steps'],
        resume_from_checkpoint=None,
        distributed_backend=None,
    )

    # logger.info(f'Zero shot results')
    # _module.extra_tag = 'zero'
    # trainer.test(_module)

    logger.info('Tuning')
    _module.extra_tag = 'fit'
    if config['train_setup']['do_train'] is True:
        trainer.fit(_module)

    logger.info('Tuned Results')
    _module.extra_tag = 'tuned'
    trainer.test(_module)


if __name__ == '__main__':
    main()
