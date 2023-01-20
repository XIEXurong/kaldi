
N=20
weight=0.5
py_nnlm_tag=pytorch_transformer
py_nnlm=exp/pytorchnn_lm/pytorch_transformer
pynnlm_config=

use_gpu=true

LM_path=data/lang_bpe_
LM=sw1_fsh_fg

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

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu $use_gpu --use-nbest true --LM_path $LM_path --LM $LM --other_opt '--gpu_wait true --stage 5 --limit_num_gpus_cmd "\"\""' \
 --decode_dir $dir --decode_dir_suffix 0graph_${py_nnlm_tag} --weight $weight --nbest_num $N --pytorch-path $py_nnlm $pynnlm_config \
 "${decode_set}" $baseline

