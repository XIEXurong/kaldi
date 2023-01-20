#!/usr/bin/env bash
# Copyright    2017  Hossein Hadian

# steps/info/chain_dir_info.pl exp/chain/e2e_tdnnf_1a
# exp/chain/e2e_tdnnf_1a: num-iters=180 nj=2..8 num-params=6.8M dim=40->84 combine=-0.060->-0.060 (over 3) logprob:train/valid[119,179,final]=(-0.080,-0.062,-0.062/-0.089,-0.083,-0.083)

set -e

# configs for 'chain'
stage=2
train_stage=-10
get_egs_stage=-10
affix=1a_ivpca3_bi_mmice_specaugkaldi_SAT1_LHUC_l1
if [ -e data/rt03 ]; then maybe_rt03=rt03; else maybe_rt03= ; fi

baseline_model=exp/chain/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi/100.mdl

decode_nj=50

# training options
dropout_schedule='0,0@0.20,0.3@0.50,0'
num_epochs=6
num_jobs_initial=3
num_jobs_final=16
minibatch_size=150=64/300=64,32/600=32,16/1200=8
common_egs_dir=
xent_regularize=0.1
l2_regularize=0.00005
frames_per_iter=1500000
cmvn_opts="--norm-means=false --norm-vars=false"
train_set=train_nodup_spe2e_hires_spk
add_opt=

param_init=0.0
adapt_lr_factor=1.0

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

lang=data/lang_e2e
treedir=exp/chain/e2e_tree_bi  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_cnn_tdnnf_${affix}

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat
  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025
  batchnorm-component name=idct-batchnorm input=idct
  spec-augment-layer name=idct-spec-augment freq-max-proportion=0.3 time-zeroed-proportion=0.00001 time-mask-max-frames=2
  combine-feature-maps-layer name=combine_inputs input=Append(idct-spec-augment, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40
  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
    cat <<EOF > $dir/configs/change.config
	input-node name=input dim=41
	
	# acoustic features
	dim-range-node name=feature1 input-node=input dim=40 dim-offset=0
	component-node name=idct component=idct input=feature1

	# speaker id
	dim-range-node name=feature2 input-node=input dim=1 dim-offset=40
EOF

    spk_num=`cat data/${train_set}/num_spk`
    lhuc_scale=2.0
    dim_tmp=2560
    dim_tmp_x2=$[$dim_tmp*2]
    layer=cnn1
    cat <<EOF >> $dir/configs/change.config	
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=true learning-rate-factor=${adapt_lr_factor}
	component-node name=LHUC.linear.$layer component=LHUC.linear.$layer input=feature2
	component name=LHUC.act.$layer type=SigmoidComponent self-repair-scale=0 dim=$dim_tmp
	component-node name=LHUC.act.$layer component=LHUC.act.$layer input=LHUC.linear.$layer
	component name=LHUC.product.$layer type=ElementwiseProductComponent output-dim=$dim_tmp input-dim=$dim_tmp_x2
	component-node name=LHUC.product.$layer component=LHUC.product.$layer input=Append($layer.relu, Scale($lhuc_scale, LHUC.act.$layer))
	component-node name=$layer.batchnorm component=$layer.batchnorm input=LHUC.product.$layer
	
EOF

    nnet3-copy --nnet-config=$dir/configs/change.config $baseline_model $dir/0.raw
fi

if [ $stage -le 4 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --use-gpu "wait" \
    --feat.online-ivector-dir exp/nnet3/ivectors_pca3_train_nodup_spe2e_max2 \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter $frames_per_iter \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.add-option="$add_opt" \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --trainer.input-model $dir/0.raw \
    --cleanup.remove-egs false \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
    
  # reset the speaker-dependent parameters
  mv $dir/final.mdl $dir/final_ori.mdl
  spk_num=`cat data/${train_set}/num_spk`
  dim_tmp=2560
  layer=cnn1
  cat <<EOF >> $dir/configs/change_final.config
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=false
EOF
  nnet3-am-copy --nnet-config=$dir/configs/change_final.config $dir/final_ori.mdl $dir/final.mdl
  
  nnet3-am-info $dir/final_ori.mdl > $dir/final_ori.mdl.info
  nnet3-am-info $dir/final.mdl > $dir/final.mdl.info
fi

if [ $stage -le 5 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_nosp_sw1_tg $dir $dir/graph_sw1_tg
fi

graph_dir=$dir/graph_sw1_tg

if [ $stage -le 6 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in eval2000 $maybe_rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" \
          --online-ivector-dir exp/nnet3/ivectors_pca3_${decode_set} \
          $graph_dir data/${decode_set}_hires_spk \
          $dir/decode_${decode_set}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_nosp_sw1_{tg,fsh_fg} data/${decode_set}_hires_spk \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

exit 0;
