#!/bin/bash

set -e

# configs for 'chain'
stage=0
stop_stage=100
train_stage=-10
get_egs_stage=-10
baseline=e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi
adapt_ivector_dir=exp/nnet3/ivectors_eval2000
test_ivector_dir=exp/nnet3/ivectors_eval2000
epoch_num=7
lr1=0.001
lr2=0.001
param_init_file=
param_std_init=-2.3
log_std=true
tied_std=true
std_update_scale=0.00625 # give a small scale if tied_std == true
tag= # any other marks
pre_out_dim=256 # 384
pre_out_layer=prefinal-l # lstm3.rp
output_chain_layer=prefinal-chain.affine # output.affine

decode_iter=
decode_nj=50
decode_opt=

# training options
minibatch_size=150=64/300=64,32/600=32,16/1200=8
remove_egs=false
xent_regularize=1
add_opt=""
mmi_scale=0
l2_regularize=0.00005
allowed_lengths_file=
chunk_left_context=
chunk_right_context=
chunk_left_context_initial=
chunk_right_context_final=
frames_per_chunk_primary=
extra_left_context=
extra_right_context=
extra_left_context_initial=
extra_right_context_final=

adapted_layer="tdnnf2"
layer_dim="160" # should be corresponding to the $adapted_layer
KL_scale="0.01"
input_config="component-node name=idct component=idct input=feature1" # cnn-tdnn, for tdnn, it can be "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))"
input_config1=
input_config2=
input_config3=
input_dim=41
prior_mean_file=  # a file or a list of files
prior_std_file=  # a file or a list of files
prior_mean="0.0" # a value or a list of values
prior_std="1.0"  # a value or a list of values
common_egs_dir=
deriv_weights_scp=
spk_count_file=
adapt_base=adaptation/LHN_e2e
adapt_model=
init_model=
LM_path=data/lang_

cuda_id="0,1,2,3,4,5,6,7,8"

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

export CUDA_VISIBLE_DEVICES=$cuda_id

adapted_layer_array=($adapted_layer)
layer_dim_array=($layer_dim)
KL_scale_array=($KL_scale)
prior_mean_file_array=($prior_mean_file)
prior_std_file_array=($prior_std_file)
prior_mean_array=($prior_mean)
prior_std_array=($prior_std)
param_init_file_array=($param_init_file)
layer_num=`echo ${#adapted_layer_array[*]}`
layer_num1=`echo ${#layer_dim_array[*]}`
layer_num2=`echo ${#KL_scale_array[*]}`

[[ "$layer_num" == "$layer_num1" && "$layer_num" == "$layer_num2" ]] || exit 1;

adapt_set=$1 # eval2000_hires_spk_sub20
label_fst_dir=$2 # label_fst_dir=exp/chain/cnn_tdnn_iv_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_fst
decode_set=$3 # eval2000_hires_spk

version=_BLHN_e2e${tag}_adaptlayer${layer_num}_epoch${epoch_num}_lr1${lr1}_lr2${lr2}

dirbase=exp/chain/${baseline}
dir=exp/chain/$adapt_base/${baseline}${version}


mkdir -p $dir
cp -r $dirbase/{configs,phones.txt,phone_lm.fst,tree,den.fst,normalization.fst,0.trans_mdl} $dir/


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

spk_count=data/${adapt_set}/spk_count
spk_num=`cat data/${adapt_set}/num_spk`
input_dim_nospk=$(awk "BEGIN{print($input_dim-1)}")

if [ ! -z $spk_count_file ]; then
    spk_count=$spk_count_file
fi

# input features
cat <<EOF > $dir/configs/change.config
	input-node name=input dim=$input_dim
	
	# acoustic features
	dim-range-node name=feature1 input-node=input dim=$input_dim_nospk dim-offset=0
	${input_config}
    ${input_config1}
    ${input_config2}
    ${input_config3}

	# speaker id
	dim-range-node name=feature2 input-node=input dim=1 dim-offset=$input_dim_nospk
	
	# frame count per speaker
	component name=BLHN.count type=LinearSelectColComponent input-dim=1 output-dim=1 col-num=$spk_num matrix=$spk_count max-change=0 learning-rate-factor=0 l2-regularize=0.00 use-natural-gradient=false
	component-node name=BLHN.count component=BLHN.count input=feature2
EOF


layer_num_minus1=$(awk "BEGIN{print($layer_num-1)}")

# adaptation in each layer
for i in `seq 0 $layer_num_minus1`; do

layer=`echo ${adapted_layer_array[i]}`
dim_tmp=`echo ${layer_dim_array[i]}`
dim_tmp2=$(awk "BEGIN{print($dim_tmp*$dim_tmp)}")
dim_tmp2_plus_dim_tmp=$(awk "BEGIN{print($dim_tmp2+$dim_tmp)}")
dim_tmp2_x4plus1=$(awk "BEGIN{print(4*$dim_tmp2+1)}")
KL=`echo ${KL_scale_array[i]}`


if [[ "$i" < "${#prior_mean_array[*]}" ]]; then
  prior_mean_tmp=`echo ${prior_mean_array[i]}`
else
  prior_mean_tmp=`echo ${prior_mean_array[-1]}`
fi
prior_mean_config="output-mean=$prior_mean_tmp output-stddev=0"
if [ ! -z $prior_mean_file ]; then
  if [[ "$i" < "${#prior_mean_file_array[*]}" ]]; then
    prior_mean_file_tmp=`echo ${prior_mean_file_array[i]}`
  else
    prior_mean_file_tmp=`echo ${prior_mean_file_array[-1]}`
  fi
  prior_mean_config="vector=$prior_mean_file_tmp"
fi
if [[ "$i" < "${#prior_std_array[*]}" ]]; then
  prior_std_tmp=`echo ${prior_std_array[i]}`
else
  prior_std_tmp=`echo ${prior_std_array[-1]}`
fi
prior_std_config="output-mean=$prior_std_tmp output-stddev=0"
if [ ! -z $prior_std_file ]; then
  if [[ "$i" < "${#prior_std_file_array[*]}" ]]; then
    prior_std_file_tmp=`echo ${prior_std_file_array[i]}`
  else
    prior_std_file_tmp=`echo ${prior_std_file_array[-1]}`
  fi
  prior_std_config="vector=$prior_std_file_tmp"
fi

# Each dim x dim dimensional transforming matrices of LHN for one speaker is implemented as a dim x dim dimensional vector
if [[ "$i" -lt "${#prior_mean_array[*]}" || "$i" -lt "${#prior_mean_file_array[*]}" || "$i" -lt "${#prior_std_array[*]}" || "$i" -lt "${#prior_std_file_array[*]}" ]]; then
  prior_layer=$layer
cat <<EOF >> $dir/configs/change.config
	component name=BLHN.prior_mean.$prior_layer type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=$dim_tmp2 $prior_mean_config
	component-node name=BLHN.prior_mean.$prior_layer component=BLHN.prior_mean.$prior_layer input=feature2
	component name=BLHN.prior_std.$prior_layer type=ConstantFunctionComponent input-dim=1 is-updatable=false output-dim=$dim_tmp2 $prior_std_config
	component-node name=BLHN.prior_std.$prior_layer component=BLHN.prior_std.$prior_layer input=feature2
EOF
fi

param_init_config=
if [ ! -z $param_init_file ]; then
  param_init_config="matrix=${param_init_file_array[i]}"
fi

cat <<EOF >> $dir/configs/change.config	
	component name=BLHN.mean.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp2 col-num=$spk_num l2-regularize=0.00 use-natural-gradient=false $param_init_config
	component-node name=BLHN.mean.$layer component=BLHN.mean.$layer input=feature2
EOF

if [[ "$tied_std" == "true" ]]; then
  tie_dim=$dim_tmp # tie all output dimensions
else
  tie_dim=$dim_tmp2
fi

cat <<EOF >> $dir/configs/change.config	
	component name=BLHN.std_ori.$layer type=LinearSelectColComponent input-dim=1 output-dim=$tie_dim col-num=$spk_num l2-regularize=0.00 param-mean=$param_std_init param-stddev=0 use-natural-gradient=false
	component-node name=BLHN.std_ori.$layer component=BLHN.std_ori.$layer input=feature2
EOF

std_ori=std_ori
if [[ "$log_std" == "true" ]]; then
  std_ori=std_exp
cat <<EOF >> $dir/configs/change.config
    component name=BLHN.std_exp.$layer type=ExpComponent dim=$tie_dim self-repair-scale=0
	component-node name=BLHN.std_exp.$layer component=BLHN.std_exp.$layer input=BLHN.std_ori.$layer
EOF
fi

cat <<EOF >> $dir/configs/change.config	
	component name=BLHN.std_scale.$layer type=NoOpComponent dim=$tie_dim backprop-scale=$std_update_scale
	component-node name=BLHN.std_scale.$layer component=BLHN.std_scale.$layer input=BLHN.$std_ori.$layer
	component name=BLHN.std.$layer type=CopyNComponent input-dim=$tie_dim output-dim=$dim_tmp2
	component-node name=BLHN.std.$layer component=BLHN.std.$layer input=BLHN.std_scale.$layer
EOF

cat <<EOF >> $dir/configs/change.config	
	component name=BLHN.vec.$layer type=BayesVecKLGaussianComponent output-dim=$dim_tmp2 input-dim=$dim_tmp2_x4plus1 KL-scale=${KL} input-frame-scale=true inv-frame-scale=true rand-per-frame=false KL-output=false test-mode=false
	component-node name=BLHN.vec.$layer component=BLHN.vec.$layer input=Append(BLHN.mean.$layer, BLHN.std.$layer, BLHN.prior_mean.$prior_layer, BLHN.prior_std.$prior_layer, BLHN.count)
	component name=BLHN.multiply.$layer type=FramewiseLinearComponent input-dim=$dim_tmp2_plus_dim_tmp output-dim=$dim_tmp feat-dim=$dim_tmp
	component-node name=BLHN.multiply.$layer component=BLHN.multiply.$layer input=Append($layer.linear,BLHN.vec.$layer)
	component-node name=$layer.affine component=$layer.affine input=BLHN.multiply.$layer
EOF

done

# use cross entropy only
cat <<EOF >> $dir/configs/change.config
	component name=no_mmi type=NoOpComponent dim=$pre_out_dim backprop-scale=$mmi_scale
	component-node name=no_mmi component=no_mmi input=${pre_out_layer}
    component-node name=$output_chain_layer component=$output_chain_layer input=no_mmi
EOF

if [ -z $adapt_model ]; then
    adapt_model=$dirbase/final.mdl
fi

nnet3-am-copy --raw --binary=false --edits="set-learning-rate-factor learning-rate-factor=0" $adapt_model - | \
 sed "s/<TestMode> F/<TestMode> T/g" | sed "s/BatchNormComponent/BatchNormTestComponent/g" | sed "s/<OrthonormalConstraint> [^ ]* /<OrthonormalConstraint> 0/g" | \
 nnet3-copy --nnet-config=$dir/configs/change.config - - | nnet3-copy --edits="remove-orphans" - $dir/0.raw

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
  
  local/chain/adaptation/train_e2e_adapt.py --stage $train_stage \
    --cmd "$train_cmd" \
    --use-gpu "wait" ${deriv_weights_scp:+ --egs.deriv-weights-scp $deriv_weights_scp} \
	${adapt_ivector_dir:+ --feat.online-ivector-dir $adapt_ivector_dir} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize $l2_regularize \
    --chain.apply-deriv-weights $apply_deriv_weights \
    --trainer.add-option="$add_opt" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "" ${allowed_lengths_file:+ --egs.allowed-lengths $allowed_lengths_file} \
    ${chunk_left_context:+ --egs.chunk-left-context $chunk_left_context} \
    ${chunk_right_context:+ --egs.chunk-right-context $chunk_right_context} \
    ${chunk_left_context_initial:+ --egs.chunk-left-context-initial $chunk_left_context_initial} \
    ${chunk_right_context_final:+ --egs.chunk-right-context-final $chunk_right_context_final} \
    --trainer.num-chunk-per-minibatch $minibatch_size \
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
    --feat-dir data/${adapt_set} \
    --fst-dir $label_fst_dir \
    --dir $dir || exit 1;

  mv $dir/final.mdl $dir/final_ori.mdl
  nnet3-am-copy --binary=false $dir/final_ori.mdl - | \
   sed "s/<TestMode> F/<TestMode> T/g" | \
   nnet3-am-copy --binary=true --edits="remove-orphans" - $dir/final.mdl

  nnet3-am-info $dir/final.mdl > $dir/final.mdl.info
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 ${LM_path}sw1_tg $dir $dir/graph_sw1_tg
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
		${LM_path}sw1_{tg,fsh_fg} data/${decode_set} \
		$dir/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;

  ) || touch $dir/.error &
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

