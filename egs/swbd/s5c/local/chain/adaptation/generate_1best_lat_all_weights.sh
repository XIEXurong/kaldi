
lm_scale=10
penalty=0.0

best_dir=1BEST_weights


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454
model=$2 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/final.mdl
ali_lat=$3 # 1BEST_lat/score_10_0.0
base_lat=$4 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

if [ -z $base_lat ]; then
  base_lat=$dir
fi

num=`cat $base_lat/num_jobs`
for i in `seq 1 $num`; do
mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty} && lattice-scale --lm-scale=$lm_scale \
 "ark:gunzip -c $base_lat/lat.$i.gz|" ark:- | \
 lattice-add-penalty --word-ins-penalty=$penalty ark:- "ark,t:|gzip -c>$dir/$best_dir/score_${lm_scale}_${penalty}/lat.$i.gz"

lattice-to-post "ark,s,cs:gunzip -c $dir/$best_dir/score_${lm_scale}_${penalty}/lat.$i.gz|" ark:- | post-to-pdf-post $model ark,s,cs:- ark:- | get-post-on-ali ark,s,cs:- \
    "ark,s,cs:gunzip -c $dir/$ali_lat/lat.$i.gz | lattice-best-path ark,s,cs:- ark:/dev/null ark:- | ali-to-pdf $model ark,s,cs:- ark:- |" \
    "ark,t,scp:$dir/$best_dir/score_${lm_scale}_${penalty}/weights.$i.ark,$dir/$best_dir/score_${lm_scale}_${penalty}/weights.$i.scp"
done

for i in `seq $num`; do
  cat $dir/$best_dir/score_${lm_scale}_${penalty}/weights.$i.scp 
done > $dir/$best_dir/score_${lm_scale}_${penalty}/weights.scp

echo $num > $dir/$best_dir/score_${lm_scale}_${penalty}/num_jobs






