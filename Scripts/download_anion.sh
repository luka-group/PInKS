mkdir -p ../Outputs/Corpora/ANION

# Download openwebtext from google drive
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WmI7SbXqIGUb8AbiF9XPwt2DfR2FBXgP' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WmI7SbXqIGUb8AbiF9XPwt2DfR2FBXgP" && rm -rf /tmp/cookies.txt
wget https://docs.google.com/uc?export=download&id=1WmI7SbXqIGUb8AbiF9XPwt2DfR2FBXgP
# ./gdown.pl https://drive.google.com/file/d/1WmI7SbXqIGUb8AbiF9XPwt2DfR2FBXgP anion_data.zip
mv anion_data.zip ../Outputs/Corpora/ANION/

# Go to the outputs folder
cd ../Outputs/Corpora/ANION/ || { echo "Failure"; exit 1; }

# Decompress corpus
echo "Decompressing anion_data"
unzip anion_data.zip

cd -  || { echo "Failure"; exit 1; }
cd ../PrepareCorpora  || { echo "Failure"; exit 1; }
python convert_anion_to_nli.py
