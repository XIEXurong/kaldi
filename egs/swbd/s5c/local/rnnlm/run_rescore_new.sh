#!/usr/bin/env bash

LM=sw1_fg_htk_arpa # using the 4-gram const arpa file as old lm
LM_path=data/lang_
decode_dir_suffix_forward=rnnlm_1e_large_drop_e40
decode_dir_suffix_backward=rnnlm_1e_back_large_drop_e40
pruned_rescore=true
ngram_order=4
weight=0.45
use_nbest=false
nbest_num=20
nbest_weight=0.8
dir_forward=exp/rnnlm/rnnlm_lstm_1e_large_drop_e40
dir_backward=exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40
forward_rescore=true
backward_rescore=true
data_dir=data
ori=_ori
lattice_prune_beam=
srun_cmd= # "srun -p pBatch_all"
num_jobs=
srun= # _mod_srun
tag=
scoring_opts=
other_opt=
decode_dir=

. ./cmd.sh
. ./utils/parse_options.sh

test_data=$1 # eval2000_fbk
ac_model_dir=$2 # exp/chain/tdnn_fbk_iv_7q

decode_dir_ori=$decode_dir

if ! $use_nbest; then

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
  
  if $forward_rescore; then
      for decode_set in $test_data; do
        if [ -z $decode_dir_ori ]; then
            decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        fi

        # Lattice rescoring
        rnnlm/lmrescore${pruned}$srun.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
          --cmd "$decode_cmd --mem 4G" ${num_jobs:+ --num-jobs $num_jobs} ${lattice_prune_beam:+ --lattice-prune-beam $lattice_prune_beam} ${srun_cmd:+ --srun-cmd $srun_cmd} \
          --weight ${weight} --max-ngram-order $ngram_order \
          ${LM_path}$LM $dir_forward \
          ${data_dir}/${decode_set} ${decode_dir} \
          ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}
          
        > ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag}/scoring_all
      done
  fi

  if $backward_rescore; then
      for decode_set in $test_data; do
        if [ -z $decode_dir_ori ]; then
            decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        fi
        if [ ! -d ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag} ]; then
          echo "$0: Must run the forward recipe first at local/rnnlm/run_tdnn_lstm.sh"
          exit 1
        fi

        # Lattice rescoring
        rnnlm/lmrescore_back$srun.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
          --cmd "$decode_cmd --mem 4G" ${srun_cmd:+ --srun-cmd $srun_cmd} \
          --weight ${weight} --max-ngram-order $ngram_order \
          ${LM_path}$LM $dir_backward \
          ${data_dir}/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_${weight}${ori}${tag} \
          ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}
        
        > ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}${ori}${tag}/scoring_all
      done
  fi

else
  
  if $forward_rescore; then
      for decode_set in $test_data; do
        if [ -z $decode_dir_ori ]; then
            decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        fi

        # Nbest rescoring
        rnnlm/lmrescore_nbest$srun.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
          --cmd "$decode_cmd --mem 4G" ${num_jobs:+ --num-jobs $num_jobs} ${srun_cmd:+ --srun-cmd $srun_cmd} \
          --N $nbest_num $nbest_weight \
          ${LM_path}$LM $dir_forward \
          ${data_dir}/${decode_set} ${decode_dir} \
          ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}
        
        > ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
      done
  fi
  
  if $backward_rescore; then
      for decode_set in $test_data; do
        if [ -z $decode_dir_ori ]; then
            decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
        fi
        if [ ! -d ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag} ]; then
          echo "$0: Must run the forward recipe first at local/rnnlm/run_tdnn_lstm.sh"
          exit 1
        fi
        
        # Nbest rescoring
        rnnlm/lmrescore_nbest_back$srun.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
          --cmd "$decode_cmd --mem 4G" ${srun_cmd:+ --srun-cmd $srun_cmd} \
          --N $nbest_num $nbest_weight \
          ${LM_path}$LM $dir_backward \
          ${data_dir}/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${nbest_weight}${tag} \
          ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}
        
        > ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
        grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${nbest_weight}${tag}/scoring_all
      done
  fi
  
fi


