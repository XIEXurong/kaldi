cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

# select per utt

matrix-average-rows scp:exp/nnet3/ivectors_eval2000/ivector_online.scp ark,t,scp:exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark,exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.scp # per utt avg

cat exp/nnet3/ivectors_train_nodup_sp/ivector_online.scp | grep "^sp1.0" | sed "s/^sp1.0-//g" > exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp.scp

matrix-average-rows scp:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp.scp ark,t,scp:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark,exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.scp

# COS
select_number=100 # begin with a large number for convinience
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perutt_cos

python3 local/chain/adaptation/ivec_select/select_ivec.py $base_feat $target_feat $select_name $select_number



# select per spk

ivector-mean ark:data/eval2000/spk2utt ark:exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark ark,t:exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark # avg all utts per spk

ivector-mean ark:data/train_nodup/spk2utt ark:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark ark,t:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark

# COS
select_number=10 # begin with a large number for convinience
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_cos

python3 local/chain/adaptation/ivec_select/select_ivec.py $base_feat $target_feat $select_name $select_number

# PROD
select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_prod

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'PROD' $base_feat $target_feat $select_name $select_number

# DIST
select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_dist

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'DIST' $base_feat $target_feat $select_name $select_number



# select per spk sub10

N=_sub10

perl local/chain/adaptation/find_pdf.pl exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark data/eval2000_hires_spk${N}/feats.scp > exp/nnet3/ivectors_eval2000/ivector_online_avg_vec${N}.ark

ivector-mean ark:data/eval2000_hires_spk${N}/spk2utt ark:exp/nnet3/ivectors_eval2000/ivector_online_avg_vec${N}.ark ark,t:exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec${N}.ark # only avg the given utts per spk

# COS
select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec${N}.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_cos${N}

python3 local/chain/adaptation/ivec_select/select_ivec.py $base_feat $target_feat $select_name $select_number

# PROD
select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec${N}.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_prod${N}

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'PROD' $base_feat $target_feat $select_name $select_number

# DIST
select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec${N}.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_dist${N}

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'DIST' $base_feat $target_feat $select_name $select_number




# aug data

cp -r data/train_nodup data/train_nodup_hires
mv data/train_nodup_hires/feats.scp data/train_nodup_hires/feats_ori.scp
cat data/train_nodup_sp_hires/feats.scp | grep "^sp1.0" | sed "s/^sp1.0-//g" > data/train_nodup_hires/feats.scp


# preparing i-vector dir

## train_nodup
dir=exp/nnet3/ivectors_train_nodup
mkdir -p $dir
cp exp/nnet3/ivectors_train_nodup_sp/final.ie.id $dir/
cp -r exp/nnet3/ivectors_train_nodup_sp/conf $dir/
cp exp/nnet3/ivectors_train_nodup_sp/ivector_period $dir/
cat exp/nnet3/ivectors_train_nodup_sp/ivector_online.scp | grep "^sp1.0" | sed "s/^sp1.0-//g" > $dir/ivector_online.scp


## eval2000+train_nodup
dir=exp/nnet3/ivectors_eval2000+train_nodup
mkdir -p $dir
cp exp/nnet3/ivectors_eval2000/final.ie.id $dir/
cp -r exp/nnet3/ivectors_eval2000/conf $dir/
cp exp/nnet3/ivectors_eval2000/ivector_period $dir/

cat exp/nnet3/ivectors_eval2000/ivector_online.scp > $dir/ivector_online_eval2000.scp
cat exp/nnet3/ivectors_train_nodup/ivector_online.scp > $dir/ivector_online_train_nodup.scp

cat $dir/ivector_online_eval2000.scp $dir/ivector_online_train_nodup.scp > $dir/ivector_online.scp



# aug feats

N=_sub10
aug_num=1
tag=_perspk_cos
select_number=10
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train${tag}${N}

cp -r data/eval2000_hires_spk${N} data/eval2000_hires_spk${N}_aug${aug_num}${tag}
mv data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats.scp data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori.scp
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires/feats.scp data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori.scp > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_ori.scp
mv data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_ori
mv data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt_ori
cat $select_name | awk -v aug_num=$aug_num '{printf $1;for(i=2;i<=aug_num+1;i++) printf " "$i; printf "\n"}' | perl utils/spk2utt_to_utt2spk.pl > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2spk_aug
cat data/train_nodup_hires/spk2utt | perl local/chain/adaptation/find_pdf_expand_stdin.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2spk_aug true > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt_aug
perl utils/spk2utt_to_utt2spk.pl < data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt_aug > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_aug
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_ori data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_aug | sort > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_ori.scp data/train_nodup_hires/feats.scp | \
  perl local/chain/adaptation/find_pdf_stdin.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_aug_not_uniq.scp
perl local/chain/adaptation/add_num_unique.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk
perl local/chain/adaptation/add_num_unique.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_aug_not_uniq.scp > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_aug.scp

cat data/train_nodup_hires/spk2utt | perl local/chain/adaptation/find_pdf_expand_stdin.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2spk_aug > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt_aug_id
perl utils/spk2utt_to_utt2spk.pl < data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt_aug_id > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_aug_id
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_ori data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_aug_id | sort > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id_not_uniq
perl local/chain/adaptation/add_num_unique.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id

feat-to-len scp:data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_aug.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/align1.pdf
( perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_ori && \
  perl local/chain/adaptation/rm_pdf.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_ori ) > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id_ori_order # keep the original order
perl local/chain/adaptation/segment2id.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_id_ori_order data/eval2000_hires_spk${N}_aug${aug_num}${tag}/align1.pdf data/eval2000_hires_spk${N}_aug${aug_num}${tag}/num_spk > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk.ark
analyze-counts --binary=false ark:data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk_count

paste-feats scp:data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats_ori_aug.scp ark:data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk.ark ark,scp:data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats.ark,data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_hires_spk${N}_aug${aug_num}${tag}
mv data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text_ori
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text_ori data/train_nodup_hires/text | \
  perl local/chain/adaptation/find_pdf_stdin.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text_not_uniq
perl local/chain/adaptation/add_num_unique.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/text
perl local/chain/adaptation/utt2spk_to_spk2utt.pl < data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk2utt
mv data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments_ori
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments_ori data/train_nodup_hires/segments | \
  perl local/chain/adaptation/find_pdf_stdin.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments_not_uniq
perl local/chain/adaptation/add_num_unique.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments_not_uniq > data/eval2000_hires_spk${N}_aug${aug_num}${tag}/segments
perl utils/data/get_utt2dur.sh data/eval2000_hires_spk${N}_aug${aug_num}${tag}



## select ivector

N=_sub10
aug_num=1
tag=_perspk_cos
select_number=10
dir=exp/nnet3/ivectors_eval2000${N}_aug${aug_num}${tag}

mkdir -p $dir
cp exp/nnet3/ivectors_eval2000/final.ie.id $dir/
cp -r exp/nnet3/ivectors_eval2000/conf $dir/
cp exp/nnet3/ivectors_eval2000/ivector_period $dir/
perl local/chain/adaptation/find_pdf.pl exp/nnet3/ivectors_eval2000+train_nodup/ivector_online.scp data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_not_uniq > $dir/ivector_online_not_uniq.scp
perl local/chain/adaptation/add_num_unique.pl $dir/ivector_online_not_uniq.scp > $dir/ivector_online.scp


# aug lat

## training set

baseline=cnn_tdnn1a_sp

. ./cmd.sh
. ./path.sh

nj=$(cat exp/tri4_lats_nodup_sp/num_jobs)
# with chain model setting
bash steps/nnet3/align_lats.sh --nj $nj --cmd "$train_cmd" --online-ivector-dir exp/nnet3/ivectors_train_nodup \
  --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' --acoustic_scale 1.0 \
  data/train_nodup_hires data/lang_sw1_fsh_fg exp/chain/${baseline} exp/chain/${baseline}/align_lats_train_nodup_sw1_fsh_fg
rm exp/chain/${baseline}/align_lats_train_nodup_sw1_fsh_fg/fsts.*.gz # save space

bash local/chain/adaptation/generate_1best_lat_all_ori.sh exp/chain/${baseline}/align_lats_train_nodup_sw1_fsh_fg

mkdir -p exp/chain/${baseline}_decode_aug
lattice-copy "ark:gunzip -c exp/chain/${baseline}/align_lats_train_nodup_sw1_fsh_fg/1BEST_lat/lat.*.gz|" ark,scp:exp/chain/${baseline}_decode_aug/lat_all_1best.ark,exp/chain/${baseline}_decode_aug/lat_all_1best.scp


## test set

N=_sub10
aug_num=1
tag=_perspk_cos
baseline=cnn_tdnn1a_sp
base_dir=exp/chain/${baseline}_decode_aug
ori_lat_dir=exp/chain/cnn_tdnn1a_sp_subN/decode_eval2000_hires_spk_sub10_sw1_fsh_fg/1BEST_lat/score_10_0.0
target_dir=$base_dir/decode_eval2000_hires_spk${N}_sw1_fsh_fg_1best_aug${aug_num}${tag}

mkdir -p $target_dir
lattice-copy "ark:gunzip -c $ori_lat_dir/lat.*.gz|" ark,scp:$target_dir/lat_ori_1best.ark,$target_dir/lat_ori_1best.scp
cat data/eval2000_hires_spk${N}_aug${aug_num}${tag}/utt2spk_aug | sort -u > $target_dir/utt2spk_aug

num=50
bash utils/split_data.sh data/eval2000_hires_spk${N}_aug${aug_num}${tag} $num

perl local/chain/adaptation/find_pdf.pl exp/chain/${baseline}_decode_aug/lat_all_1best.scp $target_dir/utt2spk_aug > $target_dir/lat_aug_1best_not_uniq.scp
perl local/chain/adaptation/add_num_unique.pl $target_dir/lat_aug_1best_not_uniq.scp > $target_dir/lat_aug_1best.scp
cat $target_dir/lat_ori_1best.scp $target_dir/lat_aug_1best.scp > $target_dir/lat_1best.scp

for i in `seq $num`; do
    subset=data/eval2000_hires_spk${N}_aug${aug_num}${tag}/split$num/$i/feats.scp
    perl local/chain/adaptation/find_pdf.pl $target_dir/lat_1best.scp $subset > $target_dir/lat_1best.$i.scp
    lattice-copy --include=$subset scp:$target_dir/lat_1best.$i.scp "ark,t:|gzip -c>$target_dir/lat.$i.gz"
done

echo $num > $target_dir/num_jobs



# weighting

N=_sub10
aug_num=1
tag=_perspk_cos
baseline=cnn_tdnn1a_sp
base_dir=exp/chain/${baseline}_decode_aug
target_dir=$base_dir/decode_eval2000_hires_spk${N}_sw1_fsh_fg_1best_aug${aug_num}${tag}

w1=1
w2=0.1

feat-to-len scp:data/eval2000_hires_spk${N}/feats.scp ark,t:- | \
 perl local/chain/adaptation/gen0ark.pl true ${w1} 3 | \
 matrix-sum-rows ark:- ark,t,scp:$target_dir/weight${w1}_ori.ark,$target_dir/weight${w1}_ori.scp

perl local/chain/adaptation/rm_pdf.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/feats.scp data/eval2000_hires_spk${N}/feats.scp > $target_dir/feat_aug.scp

feat-to-len scp:$target_dir/feat_aug.scp ark,t:- | \
 perl local/chain/adaptation/gen0ark.pl true ${w2} 3 | \
 matrix-sum-rows ark:- ark,t,scp:$target_dir/weight${w2}_aug.ark,$target_dir/weight${w2}_aug.scp

cat $target_dir/weight${w1}_ori.scp $target_dir/weight${w2}_aug.scp > $target_dir/weight${w1}_${w2}.scp


perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk data/eval2000_hires_spk${N}/feats.scp > $target_dir/spk_ori
perl local/chain/adaptation/rm_pdf.pl data/eval2000_hires_spk${N}_aug${aug_num}${tag}/spk data/eval2000_hires_spk${N}/feats.scp > $target_dir/spk_aug

analyze-counts --binary=false ark:$target_dir/spk_ori $target_dir/spk_count_ori
analyze-counts --binary=false ark:$target_dir/spk_aug $target_dir/spk_count_aug

vector-sum --binary=false \
  "vector-scale --scale=${w1} $target_dir/spk_count_ori - |" \
  "vector-scale --scale=${w2} $target_dir/spk_count_aug - |" \
  $target_dir/spk_count${w1}_${w2}


