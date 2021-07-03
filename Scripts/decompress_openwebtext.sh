find openwebtext/ -name "urlsf_subset*.xz" -type f -maxdepth 2 \
  -exec sh -c 'xz --decompress "$1"' sh {} ';'