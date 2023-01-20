
best_dir=1BEST_lat

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/tdnn_fbk_iv_7q_subN_decode/decode_eval2000_fbk_spk${split}_sw1_fg_htk_arpa

num=`cat $dir/num_jobs`
for i in `seq 1 $num`; do
mkdir -p $dir/$best_dir && lattice-1best "ark:gunzip -c $dir/lat.$i.gz|" "ark,t:|gzip -c>$dir/$best_dir/lat.$i.gz"
done
echo $num > $dir/$best_dir/num_jobs






