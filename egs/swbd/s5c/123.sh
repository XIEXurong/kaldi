#!/usr/bin/env bash

stage=0
train_discriminative=false  # by default, don't do the GMM-based discriminative
                            # training.

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


# This setup was modified from egs/swbd/s5b, with the following changes:
# 1. added more training data for early stages
# 2. removed SAT system (and later stages) on the 100k utterance training data
# 3. reduced number of LM rescoring, only sw1_tg and sw1_fsh_fg remain
# 4. mapped swbd transcription to fisher style, instead of the other way around

set -e # exit on error
has_fisher=true

if [ $stage -le 0 ]; then
  local/swbd1_data_download.sh /mnt/d/xxr/data/swbd/LDC97S62
  # local/swbd1_data_download.sh /mnt/matylda2/data/SWITCHBOARD_1R2 # BUT,
fi
