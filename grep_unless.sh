#grep -i "[\w\-\\\/\+\* ,']*unless[\w\-\\\/\+\* ,']*" /nas/home/qasemi/CQplus/Outputs/Corpora/OpenWebText/openwebtext/urlsf_subset00-100_data -a
file_path="$1"
# message='ACTION unless CONDITION.'
#pattern="[\w\-\\\/\+\* ,\']*"
# pattern="[\w]*"
# no_action="${message//ACTION/${pattern}}"
# final="${no_action//CONDITION/${pattern}}"
# echo "${final}"

grep -i -c 'unless' "$file_path"