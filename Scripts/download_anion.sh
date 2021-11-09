mkdir -p ../Outputs/Corpora/ANION

# Download openwebtext from google drive
./gdown.pl https://drive.google.com/file/d/1WmI7SbXqIGUb8AbiF9XPwt2DfR2FBXgP anion_data.zip
mv anion_data.zip ../Outputs/Corpora/ANION/

# Go to the outputs folder
cd ../Outputs/Corpora/ANION/ || { echo "Failure"; exit 1; }

# Decompress corpus
echo "Decompressing anion_data"
unzip anion_data.zip

cd -  || { echo "Failure"; exit 1; }
cd ../PrepareCorpora  || { echo "Failure"; exit 1; }
python convert_anion_to_nli.py
