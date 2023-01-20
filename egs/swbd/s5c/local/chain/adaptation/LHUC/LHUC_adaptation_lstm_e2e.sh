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
lr1=0.01
lr2=0.01
param_init=0.0
act="Sig" # Sig, Idnt, Exp
tag= # any other marks
pre_out_dim=384
pre_out_layer=lstm3.rp

decode_iter=
decode_nj=50
decode_opt=

adapt_database=data
egs_opts=

# training options
minibatch_size=150=64/300=64,32/600=32,16/1200=8
remove_egs=false
xent_regularize=1
add_opt=""
mmi_scale=0
l2_regularize=0.00005
allowed_lengths_file=
chunk_left_context=40
chunk_right_context=0
frames_per_chunk_primary=150
extra_left_context=50
extra_right_context=0

adapted_layer="cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12"
layer_dim="2560 1536 1536 1536 1536 1536 1536" # should be corresponding to the $adapted_layer
KL_scale="0.0001 1.0 1.0 1.0 1.0 1.0 1.0"
input_config="component-node name=idct component=idct input=feature1" # cnn-tdnn, for tdnn, it can be "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))"
input_config1=
input_config2=
input_config3=
input_dim=41
common_egs_dir=
deriv_weights_scp=
adapt_base=adaptation/LHUC_e2e
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
layer_num=`echo ${#adapted_layer_array[*]}`
layer_num1=`echo ${#layer_dim_array[*]}`

[[ "$layer_num" == "$layer_num1" ]] || exit 1;

adapt_set=$1 # eval2000_hires_spk_sub20
label_fst_dir=$2 # label_lat_dir=exp/chain/cnn_tdnn_iv_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0
decode_set=$3 # eval2000_hires_spk

version=_LHUC_e2e${tag}_adaptlayer${layer_num}_act${act}_epoch${epoch_num}_lr1${lr1}_lr2${lr2}

dirbase=exp/chain/${baseline}
dir=exp/chain/$adapt_base/${baseline}${version}


mkdir -p $dir
cp -r $dirbase/{configs,phones.txt,phone_lm.fst,tree,den.fst,normalization.fst,0.trans_mdl} $dir/


if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then

spk_num=`cat ${adapt_database}/${adapt_set}/num_spk`
input_dim_nospk=$(awk "BEGIN{print($input_dim-1)}")

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
EOF

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

# adaptation in each layer
for i in `seq 0 $layer_num_minus1`; do

layer=`echo ${adapted_layer_array[i]}`
dim_tmp=`echo ${layer_dim_array[i]}`
dim_tmp_x2=$(awk "BEGIN{print(2*$dim_tmp)}")

cat <<EOF >> $dir/configs/change.config	
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=false
	component-node name=LHUC.linear.$layer component=LHUC.linear.$layer input=feature2
	component name=LHUC.act.$layer $act_component dim=$dim_tmp
	component-node name=LHUC.act.$layer component=LHUC.act.$layer input=LHUC.linear.$layer
	component name=LHUC.product.$layer type=ElementwiseProductComponent output-dim=$dim_tmp input-dim=$dim_tmp_x2
	component-node name=LHUC.product.$layer component=LHUC.product.$layer input=Append($layer.relu, Scale($lhuc_scale, LHUC.act.$layer))
	component-node name=$layer.batchnorm component=$layer.batchnorm input=LHUC.product.$layer
	
EOF

done

# use cross entropy only
cat <<EOF >> $dir/configs/change.config
	component name=no_mmi type=NoOpComponent dim=$pre_out_dim backprop-scale=$mmi_scale
	component-node name=no_mmi component=no_mmi input=${pre_out_layer}
    component-node name=output.affine component=output.affine input=no_mmi
EOF

if [ -z $adapt_model ]; then
    adapt_model=$dirbase/final.mdl
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
    --egs.opts "$egs_opts" ${allowed_lengths_file:+ --egs.allowed-lengths $allowed_lengths_file} \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
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
    --feat-dir ${adapt_database}/${adapt_set} \
    --fst-dir $label_fst_dir \
    --dir $dir || exit 1;

  mv $dir/final.mdl $dir/final_ori.mdl
  nnet3-am-copy --binary=true --edits="remove-orphans" $dir/final_ori.mdl $dir/final.mdl
  
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
      --extra-left-context $extra_left_context \
      --extra-right-context $extra_right_context \
      --extra-left-context-initial 0 \
      --extra-right-context-final 0 \
      --frames-per-chunk "$frames_per_chunk_primary" \
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

