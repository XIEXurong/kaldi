
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

mkdir -p mfcc_hires_spk


## dev
# + spk

cp -r data/train_dev_hires data/train_dev_hires_spk
mv data/train_dev_hires_spk/feats.scp data/train_dev_hires_spk/feats_ori.scp
perl local/chain/adaptation/find_pdf.pl data/train_dev_hires/utt2spk data/train_dev_hires_spk/feats_ori.scp > data/train_dev_hires_spk/utt2spk

feat-to-len scp:data/train_dev_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/train_dev_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/train_dev_hires_spk/utt2spk data/train_dev_hires_spk/align1.pdf data/train_dev_hires_spk/num_spk > data/train_dev_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/train_dev_hires_spk/spk > data/train_dev_hires_spk/spk.ark
num=`cat data/train_dev_hires_spk/num_spk`
analyze-counts --binary=false ark:data/train_dev_hires_spk/spk data/train_dev_hires_spk/spk_count

paste-feats scp:data/train_dev_hires_spk/feats_ori.scp ark:data/train_dev_hires_spk/spk.ark ark,scp:mfcc_hires_spk/train_dev_hires_spk.ark,data/train_dev_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/train_dev_hires_spk



## train
# + spk

cp -r data/train_nodup_sp_hires data/train_nodup_sp_hires_spk
mv data/train_nodup_sp_hires_spk/feats.scp data/train_nodup_sp_hires_spk/feats_ori.scp

feat-to-len scp:data/train_nodup_sp_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/train_nodup_sp_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/train_nodup_sp_hires_spk/utt2spk data/train_nodup_sp_hires_spk/align1.pdf data/train_nodup_sp_hires_spk/num_spk > data/train_nodup_sp_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/train_nodup_sp_hires_spk/spk > data/train_nodup_sp_hires_spk/spk.ark
num=`cat data/train_nodup_sp_hires_spk/num_spk`
analyze-counts --binary=false ark:data/train_nodup_sp_hires_spk/spk data/train_nodup_sp_hires_spk/spk_count

paste-feats scp:data/train_nodup_sp_hires_spk/feats_ori.scp ark:data/train_nodup_sp_hires_spk/spk.ark ark,scp:mfcc_hires_spk/train_nodup_sp_hires_spk.ark,data/train_nodup_sp_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/train_nodup_sp_hires_spk





## train subset

utils/subset_data_dir.sh --first data/train_nodup_hires 4000 data/train_nodup_sub_hires


bash local/chain/decode.sh --decode_list "train_nodup_sub_hires" --ivector_list "train_nodup" cnn_tdnn1a_specaugkaldi_sp

for lm in tg fsh_fg; do
    dir=exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_train_nodup_sub_hires_sw1_${lm}
    grep WER $dir/wer_* | utils/best_wer.sh >> $dir/../scoring_all_sub
done


# + spk

cp -r data/train_nodup_sub_hires data/train_nodup_sub_hires_spk
mv data/train_nodup_sub_hires_spk/feats.scp data/train_nodup_sub_hires_spk/feats_ori.scp
perl local/chain/adaptation/find_pdf.pl data/train_nodup_sub_hires/utt2spk data/train_nodup_sub_hires_spk/feats_ori.scp > data/train_nodup_sub_hires_spk/utt2spk

feat-to-len scp:data/train_nodup_sub_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/train_nodup_sub_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/train_nodup_sub_hires_spk/utt2spk data/train_nodup_sub_hires_spk/align1.pdf data/train_nodup_sub_hires_spk/num_spk > data/train_nodup_sub_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/train_nodup_sub_hires_spk/spk > data/train_nodup_sub_hires_spk/spk.ark
num=`cat data/train_nodup_sub_hires_spk/num_spk`
analyze-counts --binary=false ark:data/train_nodup_sub_hires_spk/spk data/train_nodup_sub_hires_spk/spk_count

paste-feats scp:data/train_nodup_sub_hires_spk/feats_ori.scp ark:data/train_nodup_sub_hires_spk/spk.ark ark,scp:mfcc_hires_spk/train_nodup_sub_hires_spk.ark,data/train_nodup_sub_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/train_nodup_sub_hires_spk


# align lat

cp -r data/train_nodup_sub_hires data/train_nodup_sub
mv data/train_nodup_sub/feats.scp data/train_nodup_sub/feats_ori.scp
perl local/chain/adaptation/find_pdf.pl data/train_nodup/feats.scp data/train_nodup_sub/feats_ori.scp > data/train_nodup_sub/feats.scp
rm -r data/train_nodup_sub/split*

nj=50
steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train_nodup_sub \
data/lang exp/tri4 exp/tri4_lats_nodup_sub
rm exp/tri4_lats_nodup_sub/fsts.*.gz # save space





## train e2e
# + spk

cp -r data/train_nodup_spe2e_hires data/train_nodup_spe2e_hires_spk
mv data/train_nodup_spe2e_hires_spk/feats.scp data/train_nodup_spe2e_hires_spk/feats_ori.scp

feat-to-len scp:data/train_nodup_spe2e_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/train_nodup_spe2e_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/train_nodup_spe2e_hires_spk/utt2spk data/train_nodup_spe2e_hires_spk/align1.pdf data/train_nodup_spe2e_hires_spk/num_spk > data/train_nodup_spe2e_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/train_nodup_spe2e_hires_spk/spk > data/train_nodup_spe2e_hires_spk/spk.ark
num=`cat data/train_nodup_spe2e_hires_spk/num_spk`
analyze-counts --binary=false ark:data/train_nodup_spe2e_hires_spk/spk data/train_nodup_spe2e_hires_spk/spk_count

paste-feats scp:data/train_nodup_spe2e_hires_spk/feats_ori.scp ark:data/train_nodup_spe2e_hires_spk/spk.ark ark,scp:mfcc_hires_spk/train_nodup_spe2e_hires_spk.ark,data/train_nodup_spe2e_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/train_nodup_spe2e_hires_spk



