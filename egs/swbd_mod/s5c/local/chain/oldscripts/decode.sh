#!/bin/bash

# 7q is as 7p but a modified topology with resnet-style skip connections, more layers,
#  skinnier bottlenecks, removing the 3-way splicing and skip-layer splicing,
#  and re-tuning the learning rate and l2 regularize.  The configs are
#  standardized and substantially simplified.  There isn't any advantage in WER
#  on this setup; the advantage of this style of config is that it also works
#  well on smaller datasets, and we adopt this style here also for consistency.

# local/chain/compare_wer_general.sh --rt03 tdnn7p_sp tdnn7q_sp
# System                tdnn7p_sp tdnn7q_sp
# WER on train_dev(tg)      11.80     11.79
# WER on train_dev(fg)      10.77     10.84
# WER on eval2000(tg)        14.4      14.3
# WER on eval2000(fg)        13.0      12.9
# WER on rt03(tg)            17.5      17.6
# WER on rt03(fg)            15.3      15.2
# Final train prob         -0.057    -0.058
# Final valid prob         -0.069    -0.073
# Final train prob (xent)        -0.886    -0.894
# Final valid prob (xent)       -0.9005   -0.9106
# Num-parameters               22865188  18702628


# steps/info/chain_dir_info.pl exp/chain/tdnn7q_sp
# exp/chain/tdnn7q_sp: num-iters=394 nj=3..16 num-params=18.7M dim=40+100->6034 combine=-0.058->-0.057 (over 8) xent:train/valid[261,393,final]=(-1.20,-0.897,-0.894/-1.20,-0.919,-0.911) logprob:train/valid[261,393,final]=(-0.090,-0.059,-0.058/-0.098,-0.073,-0.073)

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=_fbk_40_iv_1a_ep6_multijobs_specmaskonline4
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=exp/chain/old_models/tdnn_fbk_40_iv_7q_ep6_sp/egs
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.3@0.50,0'

test_online_decoding=false  # if true, it will run the last decoding stage.

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

suffix=
$speed_perturb && suffix=_sp
dir=exp/chain/old_models/cnn_tdnn${affix}${suffix}



graph_dir=$dir/graph_sw1_tg
iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi


if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000_fbk_40; do
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
  
  > $dir/scoring_all
    grep Sum $dir/decode_eval2000_fbk_40_sw1_tg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/decode_eval2000_fbk_40_sw1_tg/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/decode_eval2000_fbk_40_sw1_tg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

    grep Sum $dir/decode_eval2000_fbk_40_sw1_fsh_fg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/decode_eval2000_fbk_40_sw1_fsh_fg/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/decode_eval2000_fbk_40_sw1_fsh_fg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
fi



if [ $stage -le 16 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in rt03_fbk_40; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
		  --online-ivector-dir exp/nnet3/old_models/ivectors_rt03 \
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
  
  > $dir/scoring_rt03
    grep Sum $dir/decode_rt03_fbk_40_sw1_tg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03
    grep Sum $dir/decode_rt03_fbk_40_sw1_tg/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03
    grep Sum $dir/decode_rt03_fbk_40_sw1_tg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03

    grep Sum $dir/decode_rt03_fbk_40_sw1_fsh_fg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03
    grep Sum $dir/decode_rt03_fbk_40_sw1_fsh_fg/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03
    grep Sum $dir/decode_rt03_fbk_40_sw1_fsh_fg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_rt03
fi




if [ $stage -le 17 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in rt02_fbk_40; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
		  --online-ivector-dir exp/nnet3/old_models/ivectors_rt02 \
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
  
  > $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_tg/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_tg/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_tg/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_tg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02

    grep Sum $dir/decode_rt02_fbk_40_sw1_fsh_fg/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_fsh_fg/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_fsh_fg/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
    grep Sum $dir/decode_rt02_fbk_40_sw1_fsh_fg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_rt02
fi


