#!/usr/bin/env bash
set -euo pipefail

stage=0

trainset=train_nodup


. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


utils/data/extract_wav_segments_data_dir.sh --nj 50 data/${trainset} data/${trainset}_noseg


if [ $stage -le 2 ]; then
  echo "$0: perturbing the training data to allowed lengths"
  utils/data/get_utt2dur.sh data/${trainset}_noseg  # necessary for the next command

  # 12 in the following command means the allowed lengths are spaced
  # by 12% change in length.
  utils/data/perturb_speed_to_allowed_lengths.py 12 data/${trainset}_noseg \
                                                 data/${trainset}_spe2e_hires
  cat data/${trainset}_spe2e_hires/utt2dur | \
    awk '{print $1 " " substr($1,5)}' >data/${trainset}_spe2e_hires/utt2uniq
  utils/fix_data_dir.sh data/${trainset}_spe2e_hires
fi

if [ $stage -le 3 ]; then
  echo "$0: extracting MFCC features for the training data"
  mfccdir=mfcc_hires
  steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires.conf \
                     --cmd "$train_cmd" data/${trainset}_spe2e_hires exp/make_hires/${trainset}_spe2e_hires $mfccdir
  steps/compute_cmvn_stats.sh data/${trainset}_spe2e_hires exp/make_hires/${trainset}_spe2e_hires $mfccdir
fi


# ivector with GMM trained on pca features (first 30k)
bash local/nnet3/run_ivector_common_pca.sh

max_num=2
train_set=train_nodup_spe2e
bash steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
  data/${train_set}_max${max_num}_hires exp/nnet3/extractor_pca exp/nnet3/ivectors_pca_${train_set}_max${max_num}


# ivector with GMM trained on pca features (random seleted 30k)
bash local/nnet3/run_ivector_common_pca1.sh

max_num=2
train_set=train_nodup_spe2e
bash steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
  data/${train_set}_max${max_num}_hires exp/nnet3/extractor_pca1 exp/nnet3/ivectors_pca1_${train_set}_max${max_num}

