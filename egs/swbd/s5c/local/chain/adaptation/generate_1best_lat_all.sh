
lm_scale=10
penalty=0.0

best_dir=1BEST_lat


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/tdnn_fbk_iv_7q_subN_decode/decode_eval2000_fbk_spk${split}_sw1_fg_htk_arpa

num=`cat $dir/num_jobs`
for i in `seq 1 $num`; do
mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty} && lattice-scale --lm-scale=$lm_scale \
 "ark:gunzip -c $dir/lat.$i.gz|" ark:- | \
 lattice-add-penalty --word-ins-penalty=$penalty ark:- ark:- | \
 lattice-1best ark:- "ark,t:|gzip -c>$dir/$best_dir/score_${lm_scale}_${penalty}/lat.$i.gz"
done
echo $num > $dir/$best_dir/score_${lm_scale}_${penalty}/num_jobs






