#!/bin/bash

set -e

# configs for 'chain'
stage=0
stop_stage=100
train_stage=-10
get_egs_stage=-10
baseline=cnn_tdnn_iv_1a_hires_sp
adapt_ivector_dir=exp/nnet3/ivectors_eval2000
test_ivector_dir=exp/nnet3/ivectors_eval2000
epoch_num=7
lr1=0.01
lr2=0.01
num_jobs_init=1
num_jobs_final=1
num_chunk=64
param_init=0.0
act="Sig" # Sig, Idnt, Exp
tag= # any other marks
pre_out_dim=256
alignment_subsampling_factor=1
adapt_lr_factor=1.0
constrain=false

adapt_tied=true
adapt_dim=100
adapt_input=LHUC.linear
adapt_op=

decode_iter=
decode_nj=50
decode_opt=

# training options
frames_per_eg=150,100,50,20,10,5
remove_egs=false
xent_regularize=1.0
add_opt="--optimization.memory-compression-level=2"
mmi_scale=0.0
l2_regularize=0
chunk_left_context=
chunk_right_context=
chunk_left_context_initial=
chunk_right_context_final=
deriv_truncate_margin=
frames_per_chunk_primary=
extra_left_context=
extra_right_context=
extra_left_context_initial=
extra_right_context_final=

adapted_layer="cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12"
layer_dim="2560 1536 1536 1536 1536 1536 1536" # should be corresponding to the $adapted_layer
input_config="component-node name=idct component=idct input=feature1" # cnn-tdnn, for tdnn, it can be "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))"
input_config1=
input_config2=
input_config3=
input_dim=41
common_egs_dir=
deriv_weights_scp=
adapt_base=adaptation/LHUC
adapt_model=
init_model=
latbase=

cuda_id="0,1,2,3,4,5,6,7,8"

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

export CUDA_VISIBLE_DEVICES=$cuda_id

adapted_layer_array=($adapted_layer)
layer_dim_array=($layer_dim)
layer_num=`echo ${#adapted_layer_array[*]}`
layer_num1=`echo ${#layer_dim_array[*]}`

[[ "$layer_num" == "$layer_num1" ]] || exit 1;

adapt_set=$1 # eval2000_hires_spk_sub20
label_lat_dir=$2 # label_lat_dir=exp/chain/cnn_tdnn_iv_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0
decode_set=$3 # eval2000_hires_spk

num_chunk1=`echo $num_chunk | tr "," "_"`
version=_LHUC${tag}_adaptlayer${layer_num}_act${act}_batch${num_chunk1}_epoch${epoch_num}_lr1${lr1}_lr2${lr2}

dirbase=exp/chain/${baseline}
dir=exp/chain/$adapt_base/${baseline}${version}


mkdir -p $dir
cp -r $dirbase/{configs,phones.txt,phone_lm.fst,tree,den.fst,normalization.fst,0.trans_mdl} $dir/

if [ -z $latbase ]; then
    cp -r $dirbase/{tree,phones.txt} $label_lat_dir/
else
    cp -r $latbase/{tree,phones.txt} $label_lat_dir/
fi


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

spk_num=`cat data/${adapt_set}/num_spk`
input_dim_nospk=$(awk "BEGIN{print($input_dim-1)}")

# input features
> $dir/configs/change.config

layer_num_minus1=$(awk "BEGIN{print($layer_num-1)}")

act_component="type=SigmoidComponent self-repair-scale=0" # Sig
lhuc_scale=2.0
if [[ "$act" == "Idnt" ]]; then
  act_component="type=NoOpComponent"
  lhuc_scale=1.0
elif [[ "$act" == "Exp" ]]; then
  act_component="type=ExpComponent self-repair-scale=0"
  lhuc_scale=1.0
fi

if [[ "$adapt_tied" == "true" ]]; then
cat <<EOF >> $dir/configs/change.config	
	component name=LHUC.linear.tied type=LinearSelectColComponent input-dim=1 output-dim=$adapt_dim col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=false learning-rate-factor=${adapt_lr_factor}
	component-node name=$adapt_input.tied$adapt_op component=LHUC.linear.tied input=feature2
EOF
fi

# adaptation in each layer
for i in `seq 0 $layer_num_minus1`; do

layer=`echo ${adapted_layer_array[i]}`
dim_tmp=`echo ${layer_dim_array[i]}`
dim_tmp_x2=$(awk "BEGIN{print(2*$dim_tmp)}")

if [[ "$adapt_tied" == "false" ]]; then
cat <<EOF >> $dir/configs/change.config	
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$adapt_dim col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=false learning-rate-factor=${adapt_lr_factor}
	component-node name=$adapt_input.$layer$adapt_op component=LHUC.linear.$layer input=feature2
EOF
fi

done

# use cross entropy only
cat <<EOF >> $dir/configs/change.config
	component name=no_mmi type=NoOpComponent dim=$pre_out_dim backprop-scale=$mmi_scale
	component-node name=no_mmi component=no_mmi input=prefinal-l
	component-node name=prefinal-chain.affine component=prefinal-chain.affine input=no_mmi
EOF

if [ -z $adapt_model ]; then
    adapt_model=$dirbase/final.mdl
fi

if [ -z $latbase ]; then
    cp -r $adapt_model $label_lat_dir/
else
    cp -r $latbase/final.mdl $label_lat_dir/
fi

nnet3-am-copy --raw --binary=false --edits="set-learning-rate-factor learning-rate-factor=0" $adapt_model - | \
 sed "s/<TestMode> F/<TestMode> T/g" | sed "s/BatchNormComponent/BatchNormTestComponent/g" | sed "s/<OrthonormalConstraint> [^ ]* /<OrthonormalConstraint> 0/g" | \
 nnet3-copy --nnet-config=$dir/configs/change.config - $dir/0.raw

nnet3-info $dir/0.raw > $dir/0.raw.info

fi

if [ ! -z $init_model ]; then
    nnet3-copy $init_model $dir/0.raw
    nnet3-info $dir/0.raw > $dir/0.raw.info
fi

if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then

  apply_deriv_weights=false
  if [ ! -z $deriv_weights_scp ]; then
    apply_deriv_weights=true
  fi
  
  steps/nnet3/chain/train_adapt.py --stage $train_stage \
    --cmd "$train_cmd" \
    --use-gpu "wait" ${deriv_weights_scp:+ --egs.deriv-weights-scp $deriv_weights_scp} \
	${adapt_ivector_dir:+ --feat.online-ivector-dir $adapt_ivector_dir} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --chain.lm-opts="--num-extra-lm-states=2000" \
	--chain.alignment-subsampling-factor $alignment_subsampling_factor \
    --trainer.add-option="$add_opt" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained $constrain" \
    --egs.chunk-width $frames_per_eg \
    ${chunk_left_context:+ --egs.chunk-left-context $chunk_left_context} \
    ${chunk_right_context:+ --egs.chunk-right-context $chunk_right_context} \
    ${chunk_left_context_initial:+ --egs.chunk-left-context-initial $chunk_left_context_initial} \
    ${chunk_right_context_final:+ --egs.chunk-right-context-final $chunk_right_context_final} \
    --trainer.num-chunk-per-minibatch ${num_chunk} \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $epoch_num \
    --trainer.optimization.num-jobs-initial $num_jobs_init \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $lr1 \
    --trainer.optimization.final-effective-lrate $lr2 \
	--trainer.optimization.do-final-combination false \
    --trainer.max-param-change 2.0 \
	--trainer.input-model $dir/0.raw \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${adapt_set} \
    --lat-dir $label_lat_dir \
    --dir $dir || exit 1;
	
  mv $dir/final.mdl $dir/final_ori.mdl
  nnet3-am-copy --binary=true --edits="remove-orphans" $dir/final_ori.mdl $dir/final.mdl
  
  nnet3-am-info $dir/final.mdl > $dir/final.mdl.info
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
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
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  rm $dir/.error 2>/dev/null || true
  (
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
	  --nj $decode_nj --cmd "$decode_cmd" $iter_opts $decode_opt \
      ${extra_left_context:+ --extra-left-context $extra_left_context} \
      ${extra_right_context:+ --extra-right-context $extra_right_context} \
      ${extra_left_context_initial:+ --extra-left-context-initial $extra_left_context_initial} \
      ${extra_right_context_final:+ --extra-right-context-final $extra_right_context_final} \
      ${frames_per_chunk_primary:+ --frames-per-chunk "$frames_per_chunk_primary"} \
	  ${test_ivector_dir:+ --online-ivector-dir $test_ivector_dir} \
	  $graph_dir data/${decode_set} \
	  $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
	  
	  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
		data/lang_sw1_{tg,fsh_fg} data/${decode_set} \
		$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;

  ) || touch $dir/.error &
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi


