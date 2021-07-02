mkdir -p ../Outputs/Corpora/OpenWebText

# Download openwebtext from google drive
./gdown.pl https://drive.google.com/u/0/uc?id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx&export=download openwebtext.tar.xz
mv openwebtext.tar.xz ../Outputs/Corpora/OpenWebText/

# Go to the outputs folder
cd ../Outputs/Corpora/OpenWebText/

# Decompress corpus
echo "Decompressing openwebtext.tar.xz"
tar -xvf openwebtext.tar.xz

# Decompress each section
find openwebtext/ -name "urlsf_subset*.xz" -type f -maxdepth 2 \
  -exec sh -c 'xz --decompress "$1"' sh {} ';'


