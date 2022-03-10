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
if [ -e ../s5c_new/data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi
common_egs_dir=

suffix=
$speed_perturb && suffix=_sp
dirbase=exp_kaldi_feats/chain/${baseline}
dir=exp_kaldi_feats/chain/1BEST_cnn_tdnn/BLHUC_iv/${baseline}${version}${suffix}

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ -z $decode_lat_dir ]; then
	decode_lat_dir=cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_sp/decode_eval2000_fbk_40_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.45/1BEST_lat/score_10_0.0
fi

train_set=eval2000_fbk_40_spk$splitn
label_lat_dir=exp_kaldi_feats/chain/$decode_lat_dir
spk_count=../s5c_new/data/$train_set/spk_count

mkdir -p $dir
cp -r $dirbase/{configs,phones.txt,phone_lm.fst,tree,den.fst,normalization.fst,0.trans_mdl} $dir/
cp -r $dirbase/{final.mdl,tree,phones.txt} $label_lat_dir/

if [ ! -z $common_egs_dir ]; then
ln -s $common_egs_dir ${PWD}/$dir/egs
fi

spk_num=`cat ../s5c_new/data/${train_set}/num_spk`




cat <<EOF > $dir/configs/change.config
	input-node name=input dim=41
	dim-range-node name=feature1 input-node=input dim=40 dim-offset=0
	component-node name=input-batchnorm component=input-batchnorm input=feature1

	dim-range-node name=feature2 input-node=input dim=1 dim-offset=40
	
	component name=BLHUC.count type=LinearSelectColComponent input-dim=1 output-dim=1 col-num=$spk_num matrix=$spk_count max-change=0 learning-rate-factor=0 l2-regularize=0.00 use-natural-gradient=false
	component-node name=BLHUC.count component=BLHUC.count input=feature2
	
	component name=BLHUC.prior_mean type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=2560 output-mean=0 output-stddev=0
	component-node name=BLHUC.prior_mean component=BLHUC.prior_mean input=feature2
	component name=BLHUC.prior_std type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=2560 output-mean=1 output-stddev=0
	component-node name=BLHUC.prior_std component=BLHUC.prior_std input=feature2
	
	component name=BLHUC.prior_mean_tdnn type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=1536 output-mean=0 output-stddev=0
	component-node name=BLHUC.prior_mean_tdnn component=BLHUC.prior_mean_tdnn input=feature2
	component name=BLHUC.prior_std_tdnn type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=1536 output-mean=1 output-stddev=0
	component-node name=BLHUC.prior_std_tdnn component=BLHUC.prior_std_tdnn input=feature2
	
	component name=BLHUC.mean.1 type=LinearSelectColComponent input-dim=1 output-dim=2560 col-num=$spk_num l2-regularize=0.00 param-mean=0 param-stddev=0 use-natural-gradient=false
	component-node name=BLHUC.mean.1 component=BLHUC.mean.1 input=feature2
	component name=BLHUC.std1.1 type=LinearSelectColComponent input-dim=1 output-dim=1 col-num=$spk_num l2-regularize=0.00 param-mean=1 param-stddev=0 use-natural-gradient=false
	component-node name=BLHUC.std1.1 component=BLHUC.std1.1 input=feature2
	component name=BLHUC.std2.1 type=NoOpComponent dim=1 backprop-scale=0.000390625
	component-node name=BLHUC.std2.1 component=BLHUC.std2.1 input=BLHUC.std1.1
	component name=BLHUC.std.1 type=CopyNComponent input-dim=1 output-dim=2560
	component-node name=BLHUC.std.1 component=BLHUC.std.1 input=BLHUC.std2.1
	component name=BLHUC.vec.1 type=BayesVecKLGaussianComponent output-dim=2560 input-dim=10241 KL-scale=${KL} input-frame-scale=true inv-frame-scale=true rand-per-frame=false KL-output=false test-mode=true
	component-node name=BLHUC.vec.1 component=BLHUC.vec.1 input=Append(BLHUC.mean.1, BLHUC.std.1, BLHUC.prior_mean, BLHUC.prior_std, BLHUC.count)
	component name=BLHUC.sigmoid.1 type=SigmoidComponent dim=2560 self-repair-scale=0
	component-node name=BLHUC.sigmoid.1 component=BLHUC.sigmoid.1 input=BLHUC.vec.1
	component name=product_l1 type=ElementwiseProductComponent output-dim=2560 input-dim=5120
	component-node name=product_l1 component=product_l1 input=Append(cnn1.relu, Scale(2.0, BLHUC.sigmoid.1))
	component-node name=cnn1.batchnorm component=cnn1.batchnorm input=product_l1
	
EOF

for i in `seq 2 $layer_num`; do

if [[ "$KL" < "1.0" ]]; then
KL=$(awk "BEGIN{print(10*$KL)}")
fi

if [[ "$KL" > "1.0" ]]; then
KL=1.0
fi


if [ $i -lt 7 ]; then

echo "layer == 1"

else

cat <<EOF >> $dir/configs/change.config
	component name=BLHUC.mean.$i type=LinearSelectColComponent input-dim=1 output-dim=1536 col-num=$spk_num l2-regularize=0.00 param-mean=0 param-stddev=0 use-natural-gradient=false
	component-node name=BLHUC.mean.$i component=BLHUC.mean.$i input=feature2
	component name=BLHUC.std1.$i type=LinearSelectColComponent input-dim=1 output-dim=1 col-num=$spk_num l2-regularize=0.00 param-mean=1 param-stddev=0 use-natural-gradient=false
	component-node name=BLHUC.std1.$i component=BLHUC.std1.$i input=feature2
	component name=BLHUC.std2.$i type=NoOpComponent dim=1 backprop-scale=6.51e-4
	component-node name=BLHUC.std2.$i component=BLHUC.std2.$i input=BLHUC.std1.$i
	component name=BLHUC.std.$i type=CopyNComponent input-dim=1 output-dim=1536
	component-node name=BLHUC.std.$i component=BLHUC.std.$i input=BLHUC.std2.$i
	component name=BLHUC.vec.$i type=BayesVecKLGaussianComponent output-dim=1536 input-dim=6145 KL-scale=${KL} input-frame-scale=true inv-frame-scale=true rand-per-frame=false KL-output=false test-mode=true
	component-node name=BLHUC.vec.$i component=BLHUC.vec.$i input=Append(BLHUC.mean.$i, BLHUC.std.$i, BLHUC.prior_mean_tdnn, BLHUC.prior_std_tdnn, BLHUC.count)
	component name=BLHUC.sigmoid.$i type=SigmoidComponent dim=1536 self-repair-scale=0
	component-node name=BLHUC.sigmoid.$i component=BLHUC.sigmoid.$i input=BLHUC.vec.$i
	component name=product_l$i type=ElementwiseProductComponent output-dim=1536 input-dim=3072
	component-node name=product_l$i component=product_l$i input=Append(tdnnf$i.relu, Scale(2.0, BLHUC.sigmoid.$i))
	component-node name=tdnnf$i.batchnorm component=tdnnf$i.batchnorm input=product_l$i
	
EOF

fi
done


cat <<EOF >> $dir/configs/change.config
	component name=no_mmi type=NoOpComponent dim=256 backprop-scale=0.0
	component-node name=no_mmi component=no_mmi input=prefinal-l
	component-node name=prefinal-chain.affine component=prefinal-chain.affine input=no_mmi
EOF

nnet3-am-copy --raw --binary=false --edits="set-learning-rate-factor learning-rate-factor=0" $dirbase/final.mdl - | \
 sed "s/<TestMode> F/<TestMode> T/g" | sed "s/BatchNormComponent/BatchNormTestComponent/g" | sed "s/<OrthonormalConstraint> [^ ]* /<OrthonormalConstraint> 0/g" | \
 nnet3-copy --nnet-config=$dir/configs/change.config - $dir/0.raw

nnet3-info $dir/0.raw > $dir/0.raw.info


if [ $stage -le 13 ]; then
  local/chain/train_adapt.py --stage $train_stage \
    --cmd "$train_cmd" \
	--feat.online-ivector-dir exp_kaldi_feats/nnet3/ivectors_eval2000 \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
	--chain.alignment-subsampling-factor 1 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch ${num_chunk} \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $epoch_num \
    --trainer.optimization.num-jobs-initial 1 \
    --trainer.optimization.num-jobs-final 1 \
    --trainer.optimization.initial-effective-lrate $lr1 \
    --trainer.optimization.final-effective-lrate $lr2 \
	--trainer.optimization.do-final-combination false \
    --trainer.max-param-change 2.0 \
	--trainer.input-model $dir/0.raw \
    --cleanup.remove-egs $remove_egs \
    --feat-dir ../s5c_new/data/${train_set} \
    --lat-dir $label_lat_dir \
    --dir $dir || exit 1;
fi

mv $dir/final.mdl $dir/final_ori.mdl
nnet3-am-copy --binary=false $dir/final_ori.mdl - | \
 sed "s/<TestMode> F/<TestMode> T/g" > $dir/final.mdl

nnet3-am-info $dir/final.mdl > $dir/final.mdl.info

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


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
		  --online-ivector-dir exp_kaldi_feats/nnet3/ivectors_eval2000 \
          $graph_dir ../s5c_new/data/${decode_set} \
          $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
		  
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} ../s5c_new/data/${decode_set} \
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

grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fg_arpa/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fg_arpa/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fg_arpa/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg_htk_arpa/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg_htk_arpa/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_tg_htk_arpa/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/decode_eval2000_fbk_40_spk_sw1_fsh_fg/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

