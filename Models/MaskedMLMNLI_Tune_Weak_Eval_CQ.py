import logging

import hydra
import omegaconf
import pytorch_lightning as pl
# from datasets import load_dataset

from Models.DataModules.MaskedNLIDataModule import MaskedNLIDataModule
from Models.Modules.MaskedNLIModule import MaskedNLIModule

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
    # ------------
    # data
    # ------------
    data_module = MaskedNLIDataModule(config)

    # ------------
    # model
    # ------------
    lmmodel = MaskedNLIModule(config)

    # trainer = pl.Trainer(fast_dev_run=1)
    # data_module.setup('test')
    # loader = data_module.test_dataloader()
    #
    # outputs = []
    # for i, batch in zip(range(4), loader):
    #     outputs.append(
    #         lmmodel.test_step(batch, i)
    #     )
    # IPython.embed()
    # exit()
    # out = lmmodel.test_epoch_end(outputs)

    # lmmodel.load_from_checkpoint('~/CQplus/Outputs/ModifiedLangModeling/lightning_logs/version_1/checkpoints/epoch=0-step=28305.ckpt')

    # ------------
    # training
    # ------------
    trainer = pl.Trainer(
        gradient_clip_val=0,
        gpus=config['hardware']['gpus'],
        # gpus='',
        max_epochs=1,
        min_epochs=1,
        resume_from_checkpoint=None,
        distributed_backend=None,
        accumulate_grad_batches=config['trainer_args']['accumulate_grad_batches'],
        limit_train_batches=config['trainer_args']['limit_train_batches'],
    )

    trainer.fit(lmmodel, datamodule=data_module)
    trainer.save_checkpoint(f'Checkpoint/{lmmodel.__class__.__name__}.ckpt')
    trainer.test(lmmodel, datamodule=data_module)


if __name__ == '__main__':
    main()
