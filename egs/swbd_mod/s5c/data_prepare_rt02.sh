. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


mfccdir=mfcc
x=rt02
steps/make_mfcc.sh --nj 50 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir
utils/fix_data_dir.sh data/$x


fbankdir=fbk_40
x=rt02
cp -r data/$x data/${x}_fbk_40
steps/make_fbank.sh --nj 50 --cmd "$train_cmd" --fbank-config conf/fbank_40.conf data/${x}_fbk_40 exp/make_fbank/${x}_40 $fbankdir
steps/compute_cmvn_stats.sh data/${x}_fbk_40 exp/make_fbank/${x}_40 $fbankdir
utils/fix_data_dir.sh data/${x}_fbk_40


mfccdir=mfcc_hires
x=rt02
utils/copy_data_dir.sh data/$x data/${x}_hires
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf data/${x}_hires exp/make_hires/$x $mfccdir
steps/compute_cmvn_stats.sh data/${x}_hires exp/make_hires/$x $mfccdir;
utils/fix_data_dir.sh data/${x}_hires


x=rt02
steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 data/${x}_hires exp/nnet3/extractor exp/nnet3/ivectors_$x

steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 data/${x}_hires exp/nnet3/old_models/extractor exp/nnet3/old_models/ivectors_$x


