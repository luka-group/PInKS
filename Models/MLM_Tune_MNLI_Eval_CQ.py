import logging
import sys

import hydra
import omegaconf
import pytorch_lightning as pl

from DataModules.BaseNLIDataModule import MnliTuneCqTestDataModule
from Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM
import Utils

logger = logging.getLogger(__name__)


def my_handler(type, value, tb):
    logger.exception("Uncaught exception: {0}".format(str(value)))


sys.excepthook = my_handler


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    _module = NLIModuleWithTunedLM(config)

    nli_data_module = MnliTuneCqTestDataModule(config)

    # Utils.PLModelDataTest(model=_module, data=nli_data_module).run()

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

    logger.info(f'Save the tuned model')
    trainer.save_checkpoint(f'Checkpoint/{_module.__class__.__name__}.ckpt')

    # logger.info(f'Load the tuned model')
    # _module.load_from_checkpoint(f'Checkpoint/{_module.__class__.__name__}.ckpt')
    logger.info('Results on CQ')
    _module.extra_tag = 'tuned'
    trainer.test(_module, datamodule=nli_data_module)


if __name__ == '__main__':
    main()
