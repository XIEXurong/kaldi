#!/bin/bash

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
baseline=cnn_tdnn_iv_1a_hires_sp
lhuc_baseline_mat=exp/chain/adaptation/LHUC/cnn_tdnn1a_specaugkaldi_sp_LHUC_train_nodup_sp_oracle_tri4_lat_adaptlayer7_actSig_batch64_epoch6_lr10.25_lr20.025/final_lhuc_param_
adapt_ivector_dir=exp/nnet3/ivectors_train_nodup_sp
test_ivector_dir=exp/nnet3/ivectors_eval2000
epoch_num=6
lr1=0.00025
lr2=0.000025
num_jobs_init=3
num_jobs_final=16
num_chunk=64
param_init=0.0
act="Sig" # Sig, Idnt, Exp
tag= # any other marks
adapt_lr_factor=0
constrain=false

decode_iter=
decode_nj=50

# training options
frames_per_eg=150,110,100
remove_egs=false
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

adapted_layer="cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12"
layer_dim="2560 1536 1536 1536 1536 1536 1536" # should be corresponding to the $adapted_layer
input_config="component-node name=idct component=idct input=feature1" # cnn-tdnn, for tdnn, it can be "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))"
input_dim=41
common_egs_dir=
adapt_base=adaptation/LHUC

gpu_memory_required=7000
gpu_exclusive=true
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

adapt_set=$1 # train_nodup_sp_hires_spk
label_lat_dir=$2 # exp/tri4_lats_nodup_sp
treedir=$3 # exp/chain/tri5_7d_tree_sp
decode_set=$4 # eval2000_hires_spk

version=_LHUC_SAT${tag}_adaptlayer${layer_num}_act${act}

dirbase=exp/chain/${baseline}
dir=exp/chain/$adapt_base/${baseline}${version}

if [ $stage -le 0 ]; then

mkdir -p $dir
cp -r $dirbase/{configs,phones.txt,phone_lm.fst,tree,den.fst,normalization.fst,0.trans_mdl} $dir/

spk_num=`cat data/${adapt_set}/num_spk`
input_dim_nospk=$(awk "BEGIN{print($input_dim-1)}")

# input features
cat <<EOF > $dir/configs/change.config
	input-node name=input dim=$input_dim
	
	# acoustic features
	dim-range-node name=feature1 input-node=input dim=$input_dim_nospk dim-offset=0
	${input_config}

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
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp col-num=$spk_num l2-regularize=0.00 matrix=${lhuc_baseline_mat}${layer}.txt use-natural-gradient=true learning-rate-factor=${adapt_lr_factor}
	component-node name=LHUC.linear.$layer component=LHUC.linear.$layer input=feature2
	component name=LHUC.act.$layer $act_component dim=$dim_tmp
	component-node name=LHUC.act.$layer component=LHUC.act.$layer input=LHUC.linear.$layer
	component name=LHUC.product.$layer type=ElementwiseProductComponent output-dim=$dim_tmp input-dim=$dim_tmp_x2
	component-node name=LHUC.product.$layer component=LHUC.product.$layer input=Append($layer.relu, Scale($lhuc_scale, LHUC.act.$layer))
	component-node name=$layer.batchnorm component=$layer.batchnorm input=LHUC.product.$layer
	
EOF

done

nnet3-copy --nnet-config=$dir/configs/change.config $dirbase/0.raw $dir/0.raw

nnet3-info $dir/0.raw > $dir/0.raw.info

fi

if [ $stage -le 1 ]; then
  
  use_gpu=""
  train_par="_par"
  if $gpu_exclusive; then
    train_par=""
    gpu_memory_required=
    use_gpu="wait"
  fi
  
  steps/nnet3/chain/train${train_par}.py --stage $train_stage \
    --cmd "$train_cmd" ${gpu_memory_required:+ --free-memory-required $gpu_memory_required} ${use_gpu:+ --use-gpu "$use_gpu"} \
	${adapt_ivector_dir:+ --feat.online-ivector-dir $adapt_ivector_dir} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
	--trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained $constrain" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch ${num_chunk} \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $epoch_num \
    --trainer.optimization.num-jobs-initial $num_jobs_init \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $lr1 \
    --trainer.optimization.final-effective-lrate $lr2 \
    --trainer.max-param-change 2.0 \
	--trainer.input-model $dir/0.raw \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${adapt_set} \
	--tree-dir $treedir \
    --lat-dir $label_lat_dir \
    --dir $dir || exit 1;
  
  # reset the speaker-dependent parameters
  mv $dir/final.mdl $dir/final_ori.mdl
for i in `seq 0 $layer_num_minus1`; do
layer=`echo ${adapted_layer_array[i]}`
dim_tmp=`echo ${layer_dim_array[i]}`
cat <<EOF >> $dir/configs/change_final.config
	component name=LHUC.linear.$layer type=LinearSelectColComponent input-dim=1 output-dim=$dim_tmp col-num=$spk_num l2-regularize=0.00 param-mean=$param_init param-stddev=0 use-natural-gradient=false
EOF
done
  nnet3-am-copy --nnet-config=$dir/configs/change_final.config $dir/final_ori.mdl $dir/final.mdl
  
  nnet3-am-info $dir/final_ori.mdl > $dir/final_ori.mdl.info
  nnet3-am-info $dir/final.mdl > $dir/final.mdl.info
fi

if [ $stage -le 2 ]; then
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
if [ $stage -le 3 ]; then
  rm $dir/.error 2>/dev/null || true
  (
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
	  --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
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

