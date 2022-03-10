#!/bin/bash

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=false
baseline=cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_sp
layer_num=14
epoch_num=7
lr1=0.03
lr2=0.03
num_chunk=64
tag=

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,100,50,20,10,5
remove_egs=false
xent_regularize=1.0
dropout_schedule='0,0@0.20,0.5@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

splitn=$1
KL=$2
decode_lat_dir=$3
version=_BLHUC_${layer_num}Layergap7_new_rnnlmplus${splitn}_KL${KL}_increase${tag}_batch${num_chunk}_epoch${epoch_num}_lr1${lr1}_lr2${lr2}
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi
common_egs_dir=

suffix=
$speed_perturb && suffix=_sp
dirbase=exp/chain/old_models/${baseline}
dir=exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/${baseline}${version}${suffix}



graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000_fbk_40_spk; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
		  --online-ivector-dir exp/nnet3/old_models/ivectors_eval2000 \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
		  
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/old_models/lang_sw1_{tg,fsh_fg} data/${decode_set} \
            $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;

      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

