cd ../PatternsLookup

python process_dataset_using_snorkel.py \
    username=$(whoami) \
    lf_recall_threshold=0.7