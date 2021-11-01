
mkdir -p ../Outputs/Corpora/WINOVENTI

cd ../Outputs/Corpora/WINOVENTI
wget https://raw.githubusercontent.com/commonsense-exception/commonsense-exception/main/data/winoventi_bert_large_final.tsv

cd -
cd ../PrepareCorpora
python convert_winoventi_to_nli.py