mkdir -p ../Outputs/Corpora/OpenWebText

# Download openwebtext from google drive
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx" -O openwebtext.tar.xz && rm -rf /tmp/cookies.txt
mv openwebtext.tar.xz ../Outputs/Corpora/OpenWebText/

# Go to the outputs folder
cd ../Outputs/Corpora/OpenWebText/

# Decompress corpus
echo "Decompressing openwebtext.tar.xz"
tar -xvf openwebtext.tar.xz

# Decompress each section
find openwebtext/ -name "urlsf_subset*.xz" -type f -maxdepth 2 \
  -exec sh -c 'xz --decompress "$1"' sh {} ';'


