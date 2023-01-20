#!/usr/bin/env bash

LM=sw1_fg_htk_arpa # using the 4-gram const arpa file as old lm
decode_dir_suffix_forward=rnnlm_1e
decode_dir_suffix_backward=rnnlm_1e_back
pruned_rescore=true
ngram_order=4
weight=0.45
dir_forward=exp/rnnlm_lstm_1e
dir_backward=exp/rnnlm_lstm_1e_backward
data_dir=data
ori=_ori
lattice_prune_beam=

. ./cmd.sh
. ./utils/parse_options.sh

test_data=$1 # eval2000_fbk
ac_model_dir=$2 # exp/chain/tdnn_fbk_iv_7q

if [[ "$ngram_order" == 4 ]];then
ngram_order1=
else
ngram_order1=$ngram_order
fi

  pruned=
  if $pruned_rescore; then
    pruned=_pruned
	ori=${lattice_prune_beam}
  fi
  if [ ! -z ${ngram_order1} ]; then
  ori=${ori}_${ngram_order1}
  fi
  for decode_set in $test_data; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}

    # Lattice rescoring
    rnnlm/lmrescore$pruned.sh \
      --cmd "$decode_cmd --mem 4G" ${lattice_prune_beam:+ --lattice-prune-beam $lattice_prune_beam} \
      --weight ${weight} --max-ngram-order $ngram_order \
      data/lang_$LM $dir_forward \
      ${data_dir}/${decode_set} ${decode_dir} \
      ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}
	  
	> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}/scoring_all
  done


  for decode_set in $test_data; do
    decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
    if [ ! -d ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori} ]; then
      echo "$0: Must run the forward recipe first at local/rnnlm/run_tdnn_lstm.sh"
      exit 1
    fi

    # Lattice rescoring
    rnnlm/lmrescore_back.sh \
      --cmd "$decode_cmd --mem 4G" \
      --weight ${weight} --max-ngram-order $ngram_order \
      data/lang_$LM $dir_backward \
      ${data_dir}/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori} \
      ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}
	
	> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/scoring_all
	grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}/scoring_all
  done




