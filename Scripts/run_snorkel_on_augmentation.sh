cd ../PatternsLookup

python process_dataset_using_snorkel.py \
    username=$(whoami) \
    dataset_name='augment' \
    augmentation_path='/nas/home/qasemi/CQplus/Outputs/process_dataset_using_snorkel/backup/precoditions_corpus.csv' \
    lf_recall_threshold=0.0