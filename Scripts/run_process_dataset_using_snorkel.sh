cd ../PatternsLookup

python process_dataset_using_snorkel.py \
  dataset_name="omcs" \
  hydra.run.dir="/nas/home/pkhanna/CQplus/Outputs/\${hydra.job.name}/\${dataset_name}"
  