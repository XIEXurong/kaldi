
lm_scale=10
penalty=0.0
beam=20

frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1

best_dir=1BEST_lat_realign


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_eval2000_sw1_fsh_fg
data=$2 # data/eval2000_hires
ivector=$3 # exp/nnet3/ivectors_eval2000
lang=$4 # data/lang_sw1_fsh_fg
base=$5

if [ -z $base ]; then
  base=$dir/../
fi

num=`cat $dir/num_jobs`
for i in `seq 1 $num`; do
mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty}/text && lattice-scale --lm-scale=$lm_scale \
 "ark:gunzip -c $dir/lat.$i.gz|" ark:- | \
 lattice-add-penalty --word-ins-penalty=$penalty ark:- ark:- | \
 lattice-best-path ark:- "ark,t:|int2sym.pl -f 2- $lang/words.txt > $dir/$best_dir/score_${lm_scale}_${penalty}/text/text.$i"
done
echo $num > $dir/$best_dir/score_${lm_scale}_${penalty}/num_jobs


mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty}/tmp
cp -r $data $dir/$best_dir/score_${lm_scale}_${penalty}/tmp/data
rm -r $dir/$best_dir/score_${lm_scale}_${penalty}/tmp/data/split*

cat $dir/$best_dir/score_${lm_scale}_${penalty}/text/text.* | sort > $dir/$best_dir/score_${lm_scale}_${penalty}/tmp/data/text

mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty}/lat
# with chain model setting
bash steps/nnet3/align_lats.sh --nj $num --beam $beam --cmd "$train_cmd" --online-ivector-dir $ivector \
  --frames_per_chunk $frames_per_chunk \
  --extra_left_context $extra_left_context \
  --extra_right_context $extra_right_context \
  --extra_left_context_initial $extra_left_context_initial \
  --extra_right_context_final $extra_right_context_final \
  --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' --acoustic_scale 1.0 \
  $dir/$best_dir/score_${lm_scale}_${penalty}/tmp/data $lang $base $dir/$best_dir/score_${lm_scale}_${penalty}/lat

rm -r $dir/$best_dir/score_${lm_scale}_${penalty}/lat/fsts.*.gz
rm -r $dir/$best_dir/score_${lm_scale}_${penalty}/tmp


for i in `seq 1 $num`; do
lattice-1best "ark:gunzip -c $dir/$best_dir/score_${lm_scale}_${penalty}/lat/lat.$i.gz|" "ark,t:|gzip -c>$dir/$best_dir/score_${lm_scale}_${penalty}/lat.$i.gz"
done

