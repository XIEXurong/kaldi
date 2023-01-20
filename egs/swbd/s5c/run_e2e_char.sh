#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


bash local/swbd1_prepare_char_dict.sh

bash utils/prepare_lang.sh data/local/dict_char "<unk>" data/local/lang_tmp_char data/lang_char

LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
bash utils/format_lm_sri.sh --srilm-opts "$srilm_opts" data/lang_char $LM data/local/dict_char/lexicon.txt data/lang_char_sw1_tg

LM=data/local/lm/sw1_fsh.o4g.kn.gz
bash utils/build_const_arpa_lm.sh $LM data/lang_char data/lang_char_sw1_fsh_fg




