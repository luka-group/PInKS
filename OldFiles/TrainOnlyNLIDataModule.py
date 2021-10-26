import datasets

from OldFiles.BaseNLIDataModule import BaseNLIDataModule


class DNLIDataModule(BaseNLIDataModule):

    def _load_all_datasets(self):
        train_path = test_path.parent / test_path.name.replace('test', 'train')
        all_datasets = datasets.DatasetDict({
            'train': datasets.load_dataset(
                'csv', data_files=str(train_path)
            )['train'].shuffle().select([i for i in range(250)]).rename_columns({
                'question': 'action',
                'context': 'precondition',
            }),
        })
        return all_datasets

    def _group_data_in_train_test_dev(self, columns_names):
        # eval_dataset = tokenized_datasets["validation"]
        self.train_dataset = self.all_tokenized['train'].remove_columns(columns_names['train']).rename_columns({
            'nli_label': 'labels'
        })
        self.test_dataset = self.all_tokenized['test'].remove_columns(columns_names['test']).rename_columns({
            'nli_label': 'labels'
        })
        # Not sure if it is userfull
        self.train_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
        self.test_dataset.set_format(
            type='torch',
            columns=['input_ids', 'token_type_ids', 'attention_mask', 'labels'],
            output_all_columns=True,
        )
