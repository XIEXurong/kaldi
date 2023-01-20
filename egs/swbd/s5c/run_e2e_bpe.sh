#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

. ~/anaconda3/bin/activate
conda activate espnet


bash local/swbd1_prepare_bpe_dict.sh

bash utils/prepare_lang.sh data/local/dict_bpe "<unk>" data/local/lang_tmp_bpe data/lang_bpe

LM=data/local/lm/sw1.o3g.kn.gz
srilm_opts="-subset -prune-lowprobs -unk -tolower -order 3"
bash utils/format_lm_sri.sh --srilm-opts "$srilm_opts" data/lang_bpe $LM data/local/dict_bpe/lexicon.txt data/lang_bpe_sw1_tg

LM=data/local/lm/sw1_fsh.o4g.kn.gz
bash utils/build_const_arpa_lm.sh $LM data/lang_bpe data/lang_bpe_sw1_fsh_fg




