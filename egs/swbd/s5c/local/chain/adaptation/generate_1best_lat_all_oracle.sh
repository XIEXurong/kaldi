
beam=20
nj=50
tag=

frames_per_chunk=50
extra_left_context=0
extra_right_context=0
extra_left_context_initial=-1
extra_right_context_final=-1

best_dir=1BEST_lat


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_eval2000_sw1_fsh_fg
data=$2 # data/eval2000_hires
ivector=$3 # exp/nnet3/ivectors_eval2000
lang=$4 # data/lang_sw1_fsh_fg



mkdir -p $dir/align_oracle${tag}/$best_dir/tmp
cp -r $data $dir/align_oracle${tag}/$best_dir/tmp/data
rm -r $dir/align_oracle${tag}/$best_dir/tmp/data/split*

cat $data/text | awk '{printf $1; for(n=2;n<=NF;n++) printf(" %s", tolower($n)); printf("\n")}' > $dir/align_oracle${tag}/$best_dir/tmp/data/text

mkdir -p $dir/align_oracle${tag}/lat
# with chain model setting
bash steps/nnet3/align_lats.sh --nj $nj --beam $beam --cmd "$train_cmd" --online-ivector-dir $ivector \
  --frames_per_chunk $frames_per_chunk \
  --extra_left_context $extra_left_context \
  --extra_right_context $extra_right_context \
  --extra_left_context_initial $extra_left_context_initial \
  --extra_right_context_final $extra_right_context_final \
  --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' --acoustic_scale 1.0 \
  $dir/align_oracle${tag}/$best_dir/tmp/data $lang $dir $dir/align_oracle${tag}/lat

rm -r $dir/align_oracle${tag}/lat/fsts.*.gz
rm -r $dir/align_oracle${tag}/$best_dir/tmp


for i in `seq 1 $nj`; do
lattice-1best "ark:gunzip -c $dir/align_oracle${tag}/lat/lat.$i.gz|" "ark,t:|gzip -c>$dir/align_oracle${tag}/$best_dir/lat.$i.gz"
done

echo $nj > $dir/align_oracle${tag}/$best_dir/num_jobs
