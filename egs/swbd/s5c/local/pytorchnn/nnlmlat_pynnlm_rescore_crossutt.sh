
N=20
weight=0.5
py_nnlm_tag=pytorch_crossutt100_transformer
py_nnlm=exp/pytorchnn_lm/pytorch_transformer
pynnlm_config=

use_gpu=true

seq_len=100
conv_sort_list=

LM_path=data/lang_bpe_
LM=sw1_fsh_fg

cuda_id="0,1,2,3,4,5,6,7,8"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

baseline=$1 # exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi
lang=$2 # data/lang_bpe_sw1_fsh_fg
decode_set=$3 # eval2000
nnlm_dir=$4 # decode_${decode_set}_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

dir=$baseline/$nnlm_dir

if [ ! -d ${dir}_${N}best ]; then
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set} $dir ${dir}_${N}best
fi

cp -r ${dir}_${N}best ${dir}_0graph_${py_nnlm_tag}_${N}best_${weight}
nj=`cat ${dir}_${N}best/num_jobs`
for i in `seq 1 $nj`; do
    cat ${dir}_0graph_${py_nnlm_tag}_${N}best_${weight}/archives.$i/acwt | \
      awk '{print $1,0}' > ${dir}_0graph_${py_nnlm_tag}_${N}best_${weight}/archives.$i/lmwt.nolm
done

conv_conf=
if [ ! -z $conv_sort_list ]; then
    conv_conf="--conv_sort_list $conv_sort_list"
fi

bash local/pytorchnn/run_nnlm_decode_mod_crossutt.sh --cuda_id $cuda_id --use_gpu $use_gpu --use-nbest true --LM_path $LM_path --LM $LM --other_opt '--gpu_wait true --stage 5 --limit_num_gpus_cmd "\"\""' --other_opt1 "$conv_conf" \
 --decode_dir $dir --decode_dir_suffix 0graph_${py_nnlm_tag} --weight $weight --nbest_num $N --seq_len $seq_len --reset_history true --pytorch-path $py_nnlm $pynnlm_config \
 "${decode_set}" $baseline

