import logging

import hydra
import omegaconf
import pytorch_lightning as pl

from DataModules.BaseNLIDataModule import MnliTuneCqTestDataModule
from Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModuleWithTunedLM(config)

    nli_data_module = MnliTuneCqTestDataModule(config)

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

    logger.info('Tuning on MNLI data')
    _module.extra_tag = 'fit'
    if config['train_setup']['do_train'] is True:
        trainer.fit(_module, datamodule=nli_data_module)

    trainer.save_checkpoint(f'Checkpoint/{_module.__class__.__name__}.ckpt')

    logger.info('Results on CQ')
    _module.extra_tag = 'tuned'
    trainer.test(_module, datamodule=nli_data_module)


if __name__ == '__main__':
    main()
