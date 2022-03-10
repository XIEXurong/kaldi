# beseline decode

bash local/chain/oldscripts/decode.sh --stage 17

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt02_fbk_40" exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp


dir=exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_large_drop_e40_0.456_6
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

dir=exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all



bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 4 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt02_fbk_40" exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp


dir=exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_large_drop_e40_0.454
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

dir=exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all




# adaptation

bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg

bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454



cp -r data/rt02_fbk_40 data/rt02_fbk_40_spk
mv data/rt02_fbk_40_spk/feats.scp data/rt02_fbk_40_spk/feats_ori.scp
feat-to-len scp:data/rt02_fbk_40_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/rt02_fbk_40_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/rt02_fbk_40_spk/utt2spk data/rt02_fbk_40_spk/align1.pdf data/rt02_fbk_40_spk/num_spk > data/rt02_fbk_40_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/rt02_fbk_40_spk/spk > data/rt02_fbk_40_spk/spk.ark
analyze-counts --binary=false ark:data/rt02_fbk_40_spk/spk data/rt02_fbk_40_spk/spk_count

paste-feats scp:data/rt02_fbk_40_spk/feats_ori.scp ark:data/rt02_fbk_40_spk/spk.ark ark,scp:data/rt02_fbk_40_spk/feats.ark,data/rt02_fbk_40_spk/feats.scp
steps/compute_cmvn_stats.sh data/rt02_fbk_40_spk



num_chunk=64
tag=_samllstream

bash ./local/chain/oldscripts/1BEST_cnn_tdnn/BLHUC_iv/run_tdnn_mod_fbk40_ivector_BLHUC_adaptation_diffLayergap7_new_rt02_rnnlmplus_diffamount_diffKL_increase_tag.sh --frames_per_eg "150,100,50" --tag "$tag" --num-chunk ${num_chunk} --baseline cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp --layer-num 12 --epoch-num 7 --lr1 0.01 --lr2 0.01 "" 0.0001 cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp/decode_rt02_fbk_40_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 rt02_fbk_40_spk exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rt02_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01


dir=exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rt02_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01/decode_rt02_fbk_40_spk_sw1_fsh_fg_rnnlm_1e_large_drop_e40_0.456_6
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

dir=exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rt02_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01/decode_rt02_fbk_40_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd1.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbd2.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.swbdc.filt.sys | utils/best_wer.sh >> $dir/scoring_all
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all

