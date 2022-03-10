
# select per utt

matrix-average-rows scp:exp/nnet3/ivectors_eval2000/ivector_online.scp ark,t:exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark

cat exp/nnet3/ivectors_train_nodup_sp/ivector_online.scp | grep "^sp1.0" | sed "s/^sp1.0-//g" > exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp.scp

matrix-average-rows scp:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp.scp ark,t:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark


select_number=100
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perutt_cos

python3 local/chain/adaptation/ivec_select/select_ivec.py $base_feat $target_feat $select_name $select_number



# select per spk

ivector-mean ark:data/eval2000/spk2utt ark:exp/nnet3/ivectors_eval2000/ivector_online_avg_vec.ark ark,t:exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark

ivector-mean ark:data/train_nodup/spk2utt ark:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avg_vec.ark ark,t:exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark

select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_cos

python3 local/chain/adaptation/ivec_select/select_ivec.py $base_feat $target_feat $select_name $select_number


select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_prod

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'PROD' $base_feat $target_feat $select_name $select_number


select_number=10
base_feat=exp/nnet3/ivectors_train_nodup_sp/ivector_online_nosp_avgspk_vec.ark
target_feat=exp/nnet3/ivectors_eval2000/ivector_online_avgspk_vec.ark
select_name=exp/nnet3/ivectors_eval2000/select${select_number}_from_train_perspk_dist

python3 local/chain/adaptation/ivec_select/select_ivec.py --similar_score 'DIST' $base_feat $target_feat $select_name $select_number






