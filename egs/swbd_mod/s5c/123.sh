. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/eval2000_fbk_40 data/eval2000_fbk_40_spk
mv data/eval2000_fbk_40_spk/feats.scp data/eval2000_fbk_40_spk/feats_ori.scp
feat-to-len scp:data/eval2000_fbk_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_fbk_40_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_fbk_40_spk/utt2spk data/eval2000_fbk_40_spk/align1.pdf data/eval2000_fbk_40_spk/num_spk > data/eval2000_fbk_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_fbk_40_spk/spk > data/eval2000_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:data/eval2000_fbk_40_spk/spk data/eval2000_fbk_40_spk/spk_count

paste-feats scp:data/eval2000_fbk_40_spk/feats_ori.scp ark:data/eval2000_fbk_40_spk/spk.ark ark,scp:data/eval2000_fbk_40_spk/feats.ark,data/eval2000_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_fbk_40_spk




cp -r data/rt03_fbk_40 data/rt03_fbk_40_spk
mv data/rt03_fbk_40_spk/feats.scp data/rt03_fbk_40_spk/feats_ori.scp
feat-to-len scp:data/rt03_fbk_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt03_fbk_40_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_fbk_40_spk/utt2spk data/rt03_fbk_40_spk/align1.pdf data/rt03_fbk_40_spk/num_spk > data/rt03_fbk_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_fbk_40_spk/spk > data/rt03_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:data/rt03_fbk_40_spk/spk data/rt03_fbk_40_spk/spk_count

paste-feats scp:data/rt03_fbk_40_spk/feats_ori.scp ark:data/rt03_fbk_40_spk/spk.ark ark,scp:data/rt03_fbk_40_spk/feats.ark,data/rt03_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt03_fbk_40_spk

