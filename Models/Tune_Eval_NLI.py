import logging
from typing import List

import IPython
import hydra
import omegaconf
import pytorch_lightning as pl

from Models.DataModules.BaseNLIDataModule import WeakTuneCqTestDataModule, BaseNLIDataModule, MnliTuneCqTestDataModule
from Models.DataModules.CQOnlyNLIDataModule import CQOnlyNLIDataModule
from Models.Modules.BaseNLIModule import NLIModule
from Modules.NLIModuleWithTunedLM import NLIModuleWithTunedLM
import Models.Utils as Utils

logger = logging.getLogger(__name__)


@hydra.main(config_path='../Configs/model_evaluator_config.yaml')
def main(config: omegaconf.dictconfig.DictConfig):
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
    data_class_name_list = [config.data_class] if isinstance(config.data_class, str) else config.data_class
    assert isinstance(data_class_name_list, List)

    _module = _model_class(config)
    # Utils.PLModelDataTest(model=_module, data=_data).run()
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
