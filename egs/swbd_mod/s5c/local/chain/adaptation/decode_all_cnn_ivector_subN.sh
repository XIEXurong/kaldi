LM=fg_htk_arpa
data_dir=data
exp_dir=exp
ext=_hires
sp=

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

data_set=$1 # eval2000
dir=$2 # exp/chain/tdnn_fbk_iv_7q_subN_decode
dir0=$3 # exp/chain/tdnn_fbk_iv_7q

mkdir -p $dir/configs

cp -r $dir0/{cmvn_opts,frame_subsampling_factor} $dir/

cat <<EOF > $dir/configs/change.config
	input-node name=input dim=41
	dim-range-node name=feature1 input-node=input dim=40 dim-offset=0
	component-node name=idct component=idct input=feature1
EOF

nnet3-am-copy --nnet-config=$dir/configs/change.config $dir0/final.mdl $dir/final.mdl

cp -r $dir0/graph_sw1_tg $dir/graph_sw1_tg

graph_dir=$dir/graph_sw1_tg
for decode_set in ${data_set}${ext}_spk_sub5 ${data_set}${ext}_spk_sub10 ${data_set}${ext}_spk_sub20 ${data_set}${ext}_spk_sub40; do
	(
	steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
	--nj 50 --cmd "$decode_cmd" \
	--online-ivector-dir ${exp_dir}/nnet3/ivectors_${data_set}${sp} \
	$graph_dir ${data_dir}/${decode_set} \
	$dir/decode_${decode_set}_sw1_tg;

	steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
	data/lang_sw1_{tg,${LM}} ${data_dir}/${decode_set} \
	$dir/decode_${decode_set}_sw1_{tg,${LM}};
	) &
done
wait



