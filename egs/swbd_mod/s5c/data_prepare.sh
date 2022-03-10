. ./cmd.sh
. ./path.sh
. utils/parse_options.sh


fbankdir=fbk_40
for x in train eval2000 rt03; do
    cp -r data/$x data/${x}_fbk_40
    steps/make_fbank.sh --nj 50 --cmd "$train_cmd" --fbank-config conf/fbank_40.conf data/${x}_fbk_40 exp/make_fbank/${x}_40 $fbankdir
    steps/compute_cmvn_stats.sh data/${x}_fbk_40 exp/make_fbank/${x}_40 $fbankdir
    utils/fix_data_dir.sh data/${x}_fbk_40
done


for x in train_fbk_40; do
    utils/subset_data_dir.sh --first data/${x} 4000 data/${x}_dev # 5hr 6min
    n=$[`cat data/${x}/segments | wc -l` - 4000]
    utils/subset_data_dir.sh --last data/${x} $n data/${x}_nodev

    # Now-- there are 260k utterances (313hr 23min), and we want to start the
    # monophone training on relatively short utterances (easier to align), but not
    # only the shortest ones (mostly uh-huh).  So take the 100k shortest ones, and
    # then take 30k random utterances from those (about 12hr)
    utils/subset_data_dir.sh --shortest data/${x}_nodev 100000 data/${x}_100kshort
    utils/subset_data_dir.sh data/${x}_100kshort 30000 data/${x}_30kshort

    # Take the first 100k utterances (just under half the data); we'll use
    # this for later stages of training.
    utils/subset_data_dir.sh --first data/${x}_nodev 100000 data/${x}_100k
    utils/data/remove_dup_utts.sh 200 data/${x}_100k data/${x}_100k_nodup  # 110hr

    # Finally, the full training set:
    utils/data/remove_dup_utts.sh 300 data/${x}_nodev data/${x}_nodup  # 286hr

    utils/data/get_utt2dur.sh data/${x}_nodup
    utils/data/get_reco2dur.sh data/${x}_nodup
done




