
mkdir -p ../Outputs/Corpora/WINOVENTI

cd ../Outputs/Corpora/WINOVENTI  || { echo "Failure"; exit 1; }
wget https://raw.githubusercontent.com/commonsense-exception/commonsense-exception/main/data/winoventi_bert_large_final.tsv

cd - || { echo "Failure"; exit 1; }
cd ../PrepareCorpora || { echo "Failure"; exit 1; }
python convert_winoventi_to_nli.py