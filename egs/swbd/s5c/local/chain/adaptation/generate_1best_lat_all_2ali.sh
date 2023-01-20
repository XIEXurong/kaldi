
lm_scale=10
penalty=0.0

best_dir=1BEST_lat


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_eval2000_sw1_fsh_fg
data=$2 # data/eval2000_hires
ivector=$3 # exp/nnet3/ivectors_eval2000
lang=$4 # data/lang_sw1_fsh_fg
base=$5



num=`cat $dir/$best_dir/score_${lm_scale}_${penalty}/num_jobs`
for i in `seq 1 $num`; do
  lattice-best-path "ark:gunzip -c $dir/$best_dir/score_${lm_scale}_${penalty}/lat.$i.gz |" \
    "ark,t:|int2sym.pl -f 2- $dir/../graph_sw1_tg/words.txt > $dir/$best_dir/score_${lm_scale}_${penalty}/text.$i" \
    "ark,t:|gzip -c>$dir/$best_dir/score_${lm_scale}_${penalty}/ali.$i.gz"
done
