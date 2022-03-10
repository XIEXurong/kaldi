
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/eval2000_hires data/eval2000_hires_spk
mv data/eval2000_hires_spk/feats.scp data/eval2000_hires_spk/feats_ori.scp
feat-to-len scp:data/eval2000_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_hires_spk/utt2spk data/eval2000_hires_spk/align1.pdf data/eval2000_hires_spk/num_spk > data/eval2000_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_hires_spk/spk > data/eval2000_hires_spk/spk.ark
analyze-counts --binary=false ark:data/eval2000_hires_spk/spk data/eval2000_hires_spk/spk_count

paste-feats scp:data/eval2000_hires_spk/feats_ori.scp ark:data/eval2000_hires_spk/spk.ark ark,scp:data/eval2000_hires_spk/feats.ark,data/eval2000_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_hires_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/eval2000_hires data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/feats.scp data/eval2000_hires_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/eval2000_hires_spk/utt2spk $N | grep "_sub1$" > data/eval2000_hires_spk_sub${N}/utt2spk
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk/align1.pdf data/eval2000_hires_spk_sub${N}/utt2spk > data/eval2000_hires_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_hires_spk_sub${N}/utt2spk data/eval2000_hires_spk_sub${N}/align1.pdf data/eval2000_hires_spk_sub${N}/num_spk > data/eval2000_hires_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_hires_spk_sub${N}/spk > data/eval2000_hires_spk_sub${N}/spk.ark
num=`cat data/eval2000_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/eval2000_hires_spk_sub${N}/spk data/eval2000_hires_spk_sub${N}/spk_count

paste-feats scp:data/eval2000_hires_spk_sub${N}/feats_ori.scp ark:data/eval2000_hires_spk_sub${N}/spk.ark ark,scp:data/eval2000_hires_spk_sub${N}/feats.ark,data/eval2000_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/text data/eval2000_hires_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk_sub${N}/text_all data/eval2000_hires_spk_sub${N}/feats.scp > data/eval2000_hires_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/utt2dur data/eval2000_hires_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk_sub${N}/utt2dur_all data/eval2000_hires_spk_sub${N}/feats.scp > data/eval2000_hires_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/eval2000_sp_hires data/eval2000_sp_hires_spk
mv data/eval2000_sp_hires_spk/feats.scp data/eval2000_sp_hires_spk/feats_ori.scp
feat-to-len scp:data/eval2000_sp_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_sp_hires_spk/align1.pdf
cat data/eval2000_sp_hires_spk/utt2spk | sed "s/ sp.\..-/ /g" > data/eval2000_sp_hires_spk/utt2spk_all
perl local/chain/adaptation/segment2id.pl data/eval2000_sp_hires_spk/utt2spk_all data/eval2000_sp_hires_spk/align1.pdf data/eval2000_sp_hires_spk/num_spk > data/eval2000_sp_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_sp_hires_spk/spk > data/eval2000_sp_hires_spk/spk.ark
analyze-counts --binary=false ark:data/eval2000_sp_hires_spk/spk data/eval2000_sp_hires_spk/spk_count

paste-feats scp:data/eval2000_sp_hires_spk/feats_ori.scp ark:data/eval2000_sp_hires_spk/spk.ark ark,scp:data/eval2000_sp_hires_spk/feats.ark,data/eval2000_sp_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_sp_hires_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/eval2000_sp_hires data/eval2000_sp_hires_spk_sub${N}
mv data/eval2000_sp_hires_spk_sub${N}/feats.scp data/eval2000_sp_hires_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/eval2000_sp_hires_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > data/eval2000_sp_hires_spk_sub${N}/utt2spk_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_sp_hires_spk/align1.pdf data/eval2000_sp_hires_spk_sub${N}/utt2spk_all > data/eval2000_sp_hires_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_sp_hires_spk_sub${N}/utt2spk_all data/eval2000_sp_hires_spk_sub${N}/align1.pdf data/eval2000_sp_hires_spk_sub${N}/num_spk > data/eval2000_sp_hires_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_sp_hires_spk_sub${N}/spk > data/eval2000_sp_hires_spk_sub${N}/spk.ark
num=`cat data/eval2000_sp_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/eval2000_sp_hires_spk_sub${N}/spk data/eval2000_sp_hires_spk_sub${N}/spk_count

paste-feats scp:data/eval2000_sp_hires_spk_sub${N}/feats_ori.scp ark:data/eval2000_sp_hires_spk_sub${N}/spk.ark ark,scp:data/eval2000_sp_hires_spk_sub${N}/feats.ark,data/eval2000_sp_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_sp_hires_spk_sub${N}
mv data/eval2000_sp_hires_spk_sub${N}/text data/eval2000_sp_hires_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_sp_hires_spk_sub${N}/text_all data/eval2000_sp_hires_spk_sub${N}/feats.scp > data/eval2000_sp_hires_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/eval2000_sp_hires_spk_sub${N}
mv data/eval2000_sp_hires_spk_sub${N}/utt2dur data/eval2000_sp_hires_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_sp_hires_spk_sub${N}/utt2dur_all data/eval2000_sp_hires_spk_sub${N}/feats.scp > data/eval2000_sp_hires_spk_sub${N}/utt2dur

done





. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/rt03_hires data/rt03_hires_spk
mv data/rt03_hires_spk/feats.scp data/rt03_hires_spk/feats_ori.scp
feat-to-len scp:data/rt03_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt03_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_hires_spk/utt2spk data/rt03_hires_spk/align1.pdf data/rt03_hires_spk/num_spk > data/rt03_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_hires_spk/spk > data/rt03_hires_spk/spk.ark
analyze-counts --binary=false ark:data/rt03_hires_spk/spk data/rt03_hires_spk/spk_count

paste-feats scp:data/rt03_hires_spk/feats_ori.scp ark:data/rt03_hires_spk/spk.ark ark,scp:data/rt03_hires_spk/feats.ark,data/rt03_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt03_hires_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/rt03_hires data/rt03_hires_spk_sub${N}
mv data/rt03_hires_spk_sub${N}/feats.scp data/rt03_hires_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/rt03_hires_spk/utt2spk $N | grep "_sub1$" > data/rt03_hires_spk_sub${N}/utt2spk
perl local/chain/adaptation/find_pdf.pl data/rt03_hires_spk/align1.pdf data/rt03_hires_spk_sub${N}/utt2spk > data/rt03_hires_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_hires_spk_sub${N}/utt2spk data/rt03_hires_spk_sub${N}/align1.pdf data/rt03_hires_spk_sub${N}/num_spk > data/rt03_hires_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_hires_spk_sub${N}/spk > data/rt03_hires_spk_sub${N}/spk.ark
num=`cat data/rt03_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/rt03_hires_spk_sub${N}/spk data/rt03_hires_spk_sub${N}/spk_count

paste-feats scp:data/rt03_hires_spk_sub${N}/feats_ori.scp ark:data/rt03_hires_spk_sub${N}/spk.ark ark,scp:data/rt03_hires_spk_sub${N}/feats.ark,data/rt03_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/rt03_hires_spk_sub${N}
mv data/rt03_hires_spk_sub${N}/text data/rt03_hires_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/rt03_hires_spk_sub${N}/text_all data/rt03_hires_spk_sub${N}/feats.scp > data/rt03_hires_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/rt03_hires_spk_sub${N}
mv data/rt03_hires_spk_sub${N}/utt2dur data/rt03_hires_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/rt03_hires_spk_sub${N}/utt2dur_all data/rt03_hires_spk_sub${N}/feats.scp > data/rt03_hires_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/rt03_sp_hires data/rt03_sp_hires_spk
mv data/rt03_sp_hires_spk/feats.scp data/rt03_sp_hires_spk/feats_ori.scp
feat-to-len scp:data/rt03_sp_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt03_sp_hires_spk/align1.pdf
cat data/rt03_sp_hires_spk/utt2spk | sed "s/ sp.\..-/ /g" > data/rt03_sp_hires_spk/utt2spk_all
perl local/chain/adaptation/segment2id.pl data/rt03_sp_hires_spk/utt2spk_all data/rt03_sp_hires_spk/align1.pdf data/rt03_sp_hires_spk/num_spk > data/rt03_sp_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_sp_hires_spk/spk > data/rt03_sp_hires_spk/spk.ark
analyze-counts --binary=false ark:data/rt03_sp_hires_spk/spk data/rt03_sp_hires_spk/spk_count

paste-feats scp:data/rt03_sp_hires_spk/feats_ori.scp ark:data/rt03_sp_hires_spk/spk.ark ark,scp:data/rt03_sp_hires_spk/feats.ark,data/rt03_sp_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt03_sp_hires_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/rt03_sp_hires data/rt03_sp_hires_spk_sub${N}
mv data/rt03_sp_hires_spk_sub${N}/feats.scp data/rt03_sp_hires_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/rt03_sp_hires_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > data/rt03_sp_hires_spk_sub${N}/utt2spk_all
perl local/chain/adaptation/find_pdf.pl data/rt03_sp_hires_spk/align1.pdf data/rt03_sp_hires_spk_sub${N}/utt2spk_all > data/rt03_sp_hires_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_sp_hires_spk_sub${N}/utt2spk_all data/rt03_sp_hires_spk_sub${N}/align1.pdf data/rt03_sp_hires_spk_sub${N}/num_spk > data/rt03_sp_hires_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_sp_hires_spk_sub${N}/spk > data/rt03_sp_hires_spk_sub${N}/spk.ark
num=`cat data/rt03_sp_hires_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/rt03_sp_hires_spk_sub${N}/spk data/rt03_sp_hires_spk_sub${N}/spk_count

paste-feats scp:data/rt03_sp_hires_spk_sub${N}/feats_ori.scp ark:data/rt03_sp_hires_spk_sub${N}/spk.ark ark,scp:data/rt03_sp_hires_spk_sub${N}/feats.ark,data/rt03_sp_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/rt03_sp_hires_spk_sub${N}
mv data/rt03_sp_hires_spk_sub${N}/text data/rt03_sp_hires_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/rt03_sp_hires_spk_sub${N}/text_all data/rt03_sp_hires_spk_sub${N}/feats.scp > data/rt03_sp_hires_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/rt03_sp_hires_spk_sub${N}
mv data/rt03_sp_hires_spk_sub${N}/utt2dur data/rt03_sp_hires_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/rt03_sp_hires_spk_sub${N}/utt2dur_all data/rt03_sp_hires_spk_sub${N}/feats.scp > data/rt03_sp_hires_spk_sub${N}/utt2dur

done







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



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/eval2000_fbk_40 data/eval2000_fbk_40_spk_sub${N}
mv data/eval2000_fbk_40_spk_sub${N}/feats.scp data/eval2000_fbk_40_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/eval2000_fbk_40_spk/utt2spk $N | grep "_sub1$" > data/eval2000_fbk_40_spk_sub${N}/utt2spk
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_40_spk/align1.pdf data/eval2000_fbk_40_spk_sub${N}/utt2spk > data/eval2000_fbk_40_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_fbk_40_spk_sub${N}/utt2spk data/eval2000_fbk_40_spk_sub${N}/align1.pdf data/eval2000_fbk_40_spk_sub${N}/num_spk > data/eval2000_fbk_40_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_fbk_40_spk_sub${N}/spk > data/eval2000_fbk_40_spk_sub${N}/spk.ark
num=`cat data/eval2000_fbk_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/eval2000_fbk_40_spk_sub${N}/spk data/eval2000_fbk_40_spk_sub${N}/spk_count

paste-feats scp:data/eval2000_fbk_40_spk_sub${N}/feats_ori.scp ark:data/eval2000_fbk_40_spk_sub${N}/spk.ark ark,scp:data/eval2000_fbk_40_spk_sub${N}/feats.ark,data/eval2000_fbk_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_fbk_40_spk_sub${N}
mv data/eval2000_fbk_40_spk_sub${N}/text data/eval2000_fbk_40_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_40_spk_sub${N}/text_all data/eval2000_fbk_40_spk_sub${N}/feats.scp > data/eval2000_fbk_40_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/eval2000_fbk_40_spk_sub${N}
mv data/eval2000_fbk_40_spk_sub${N}/utt2dur data/eval2000_fbk_40_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_40_spk_sub${N}/utt2dur_all data/eval2000_fbk_40_spk_sub${N}/feats.scp > data/eval2000_fbk_40_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/eval2000_fbk_sp_40 data/eval2000_fbk_sp_40_spk
mv data/eval2000_fbk_sp_40_spk/feats.scp data/eval2000_fbk_sp_40_spk/feats_ori.scp
feat-to-len scp:data/eval2000_fbk_sp_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_fbk_sp_40_spk/align1.pdf
cat data/eval2000_fbk_sp_40_spk/utt2spk | sed "s/ sp.\..-/ /g" > data/eval2000_fbk_sp_40_spk/utt2spk_all
perl local/chain/adaptation/segment2id.pl data/eval2000_fbk_sp_40_spk/utt2spk_all data/eval2000_fbk_sp_40_spk/align1.pdf data/eval2000_fbk_sp_40_spk/num_spk > data/eval2000_fbk_sp_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_fbk_sp_40_spk/spk > data/eval2000_fbk_sp_40_spk/spk.ark
analyze-counts --binary=false ark:data/eval2000_fbk_sp_40_spk/spk data/eval2000_fbk_sp_40_spk/spk_count

paste-feats scp:data/eval2000_fbk_sp_40_spk/feats_ori.scp ark:data/eval2000_fbk_sp_40_spk/spk.ark ark,scp:data/eval2000_fbk_sp_40_spk/feats.ark,data/eval2000_fbk_sp_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_fbk_sp_40_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/eval2000_fbk_sp_40 data/eval2000_fbk_sp_40_spk_sub${N}
mv data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp data/eval2000_fbk_sp_40_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/eval2000_fbk_sp_40_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_sp_40_spk/align1.pdf data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all > data/eval2000_fbk_sp_40_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_fbk_sp_40_spk_sub${N}/utt2spk_all data/eval2000_fbk_sp_40_spk_sub${N}/align1.pdf data/eval2000_fbk_sp_40_spk_sub${N}/num_spk > data/eval2000_fbk_sp_40_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_fbk_sp_40_spk_sub${N}/spk > data/eval2000_fbk_sp_40_spk_sub${N}/spk.ark
num=`cat data/eval2000_fbk_sp_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/eval2000_fbk_sp_40_spk_sub${N}/spk data/eval2000_fbk_sp_40_spk_sub${N}/spk_count

paste-feats scp:data/eval2000_fbk_sp_40_spk_sub${N}/feats_ori.scp ark:data/eval2000_fbk_sp_40_spk_sub${N}/spk.ark ark,scp:data/eval2000_fbk_sp_40_spk_sub${N}/feats.ark,data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_fbk_sp_40_spk_sub${N}
mv data/eval2000_fbk_sp_40_spk_sub${N}/text data/eval2000_fbk_sp_40_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_sp_40_spk_sub${N}/text_all data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp > data/eval2000_fbk_sp_40_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/eval2000_fbk_sp_40_spk_sub${N}
mv data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur_all data/eval2000_fbk_sp_40_spk_sub${N}/feats.scp > data/eval2000_fbk_sp_40_spk_sub${N}/utt2dur

done





. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/rt03_fbk_40 data/rt03_fbk_40_spk
mv data/rt03_fbk_40_spk/feats.scp data/rt03_fbk_40_spk/feats_ori.scp
feat-to-len scp:data/rt03_fbk_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt03_fbk_40_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_fbk_40_spk/utt2spk data/rt03_fbk_40_spk/align1.pdf data/rt03_fbk_40_spk/num_spk > data/rt03_fbk_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_fbk_40_spk/spk > data/rt03_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:data/rt03_fbk_40_spk/spk data/rt03_fbk_40_spk/spk_count

paste-feats scp:data/rt03_fbk_40_spk/feats_ori.scp ark:data/rt03_fbk_40_spk/spk.ark ark,scp:data/rt03_fbk_40_spk/feats.ark,data/rt03_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt03_fbk_40_spk



. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/rt03_fbk_40 data/rt03_fbk_40_spk_sub${N}
mv data/rt03_fbk_40_spk_sub${N}/feats.scp data/rt03_fbk_40_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/rt03_fbk_40_spk/utt2spk $N | grep "_sub1$" > data/rt03_fbk_40_spk_sub${N}/utt2spk
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_40_spk/align1.pdf data/rt03_fbk_40_spk_sub${N}/utt2spk > data/rt03_fbk_40_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_fbk_40_spk_sub${N}/utt2spk data/rt03_fbk_40_spk_sub${N}/align1.pdf data/rt03_fbk_40_spk_sub${N}/num_spk > data/rt03_fbk_40_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_fbk_40_spk_sub${N}/spk > data/rt03_fbk_40_spk_sub${N}/spk.ark
num=`cat data/rt03_fbk_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/rt03_fbk_40_spk_sub${N}/spk data/rt03_fbk_40_spk_sub${N}/spk_count

paste-feats scp:data/rt03_fbk_40_spk_sub${N}/feats_ori.scp ark:data/rt03_fbk_40_spk_sub${N}/spk.ark ark,scp:data/rt03_fbk_40_spk_sub${N}/feats.ark,data/rt03_fbk_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/rt03_fbk_40_spk_sub${N}
mv data/rt03_fbk_40_spk_sub${N}/text data/rt03_fbk_40_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_40_spk_sub${N}/text_all data/rt03_fbk_40_spk_sub${N}/feats.scp > data/rt03_fbk_40_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/rt03_fbk_40_spk_sub${N}
mv data/rt03_fbk_40_spk_sub${N}/utt2dur data/rt03_fbk_40_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_40_spk_sub${N}/utt2dur_all data/rt03_fbk_40_spk_sub${N}/feats.scp > data/rt03_fbk_40_spk_sub${N}/utt2dur

done




. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

cp -r data/rt03_fbk_sp_40 data/rt03_fbk_sp_40_spk
mv data/rt03_fbk_sp_40_spk/feats.scp data/rt03_fbk_sp_40_spk/feats_ori.scp
feat-to-len scp:data/rt03_fbk_sp_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt03_fbk_sp_40_spk/align1.pdf
cat data/rt03_fbk_sp_40_spk/utt2spk | sed "s/ sp.\..-/ /g" > data/rt03_fbk_sp_40_spk/utt2spk_all
perl local/chain/adaptation/segment2id.pl data/rt03_fbk_sp_40_spk/utt2spk_all data/rt03_fbk_sp_40_spk/align1.pdf data/rt03_fbk_sp_40_spk/num_spk > data/rt03_fbk_sp_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_fbk_sp_40_spk/spk > data/rt03_fbk_sp_40_spk/spk.ark
analyze-counts --binary=false ark:data/rt03_fbk_sp_40_spk/spk data/rt03_fbk_sp_40_spk/spk_count

paste-feats scp:data/rt03_fbk_sp_40_spk/feats_ori.scp ark:data/rt03_fbk_sp_40_spk/spk.ark ark,scp:data/rt03_fbk_sp_40_spk/feats.ark,data/rt03_fbk_sp_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt03_fbk_sp_40_spk


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

for N in {5,10,20,40}; do

cp -r data/rt03_fbk_sp_40 data/rt03_fbk_sp_40_spk_sub${N}
mv data/rt03_fbk_sp_40_spk_sub${N}/feats.scp data/rt03_fbk_sp_40_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/rt03_fbk_sp_40_spk/utt2spk $N | grep "_sub1$" | sed "s/ sp.\..-/ /g" > data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_sp_40_spk/align1.pdf data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all > data/rt03_fbk_sp_40_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt03_fbk_sp_40_spk_sub${N}/utt2spk_all data/rt03_fbk_sp_40_spk_sub${N}/align1.pdf data/rt03_fbk_sp_40_spk_sub${N}/num_spk > data/rt03_fbk_sp_40_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt03_fbk_sp_40_spk_sub${N}/spk > data/rt03_fbk_sp_40_spk_sub${N}/spk.ark
num=`cat data/rt03_fbk_sp_40_spk_sub${N}/num_spk`
analyze-counts --binary=false ark:data/rt03_fbk_sp_40_spk_sub${N}/spk data/rt03_fbk_sp_40_spk_sub${N}/spk_count

paste-feats scp:data/rt03_fbk_sp_40_spk_sub${N}/feats_ori.scp ark:data/rt03_fbk_sp_40_spk_sub${N}/spk.ark ark,scp:data/rt03_fbk_sp_40_spk_sub${N}/feats.ark,data/rt03_fbk_sp_40_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/rt03_fbk_sp_40_spk_sub${N}
mv data/rt03_fbk_sp_40_spk_sub${N}/text data/rt03_fbk_sp_40_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_sp_40_spk_sub${N}/text_all data/rt03_fbk_sp_40_spk_sub${N}/feats.scp > data/rt03_fbk_sp_40_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/rt03_fbk_sp_40_spk_sub${N}
mv data/rt03_fbk_sp_40_spk_sub${N}/utt2dur data/rt03_fbk_sp_40_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/rt03_fbk_sp_40_spk_sub${N}/utt2dur_all data/rt03_fbk_sp_40_spk_sub${N}/feats.scp > data/rt03_fbk_sp_40_spk_sub${N}/utt2dur

done

