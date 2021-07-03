cd ../Outputs/Corpora/OpenWebText/
find openwebtext/ -maxdepth 2 -name "urlsf_subset*.xz" -type f \
  -exec sh -c 'xz --decompress "$1"' sh {} ';'
