cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnnf_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



##################################################

# add len

trainset=eval2000

. ./cmd.sh
. ./path.sh

utils/data/extract_wav_segments_data_dir.sh --nj 50 data/${trainset} data/${trainset}_noseg
utils/data/get_utt2dur.sh data/${trainset}_noseg
utils/data/perturb_speed_to_allowed_lengths.py --speed-perturb false --coverage-factor 0 12 data/${trainset}_noseg data/${trainset}_e2e_hires
utils/fix_data_dir.sh data/${trainset}_e2e_hires

mfccdir=mfcc_hires
steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" data/${trainset}_e2e_hires exp/make_hires/${trainset}_e2e_hires $mfccdir
steps/compute_cmvn_stats.sh data/${trainset}_e2e_hires exp/make_hires/${trainset}_e2e_hires $mfccdir

cp -r data/${trainset}_e2e_hires data/${trainset}_e2e_hires_spk
mv data/${trainset}_e2e_hires_spk/feats.scp data/${trainset}_e2e_hires_spk/feats_ori.scp
feat-to-len scp:data/${trainset}_e2e_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/${trainset}_e2e_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/${trainset}_e2e_hires_spk/utt2spk data/${trainset}_e2e_hires_spk/align1.pdf data/${trainset}_e2e_hires_spk/num_spk > data/${trainset}_e2e_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/${trainset}_e2e_hires_spk/spk > data/${trainset}_e2e_hires_spk/spk.ark
analyze-counts --binary=false ark:data/${trainset}_e2e_hires_spk/spk data/${trainset}_e2e_hires_spk/spk_count

paste-feats scp:data/${trainset}_e2e_hires_spk/feats_ori.scp ark:data/${trainset}_e2e_hires_spk/spk.ark ark,scp:data/${trainset}_e2e_hires_spk/feats.ark,data/${trainset}_e2e_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/${trainset}_e2e_hires_spk



bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01




bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1





trainset=rt03

. ./cmd.sh
. ./path.sh

utils/data/extract_wav_segments_data_dir.sh --nj 50 data/${trainset} data/${trainset}_noseg
utils/data/get_utt2dur.sh data/${trainset}_noseg
utils/data/perturb_speed_to_allowed_lengths.py --speed-perturb false --coverage-factor 0 12 data/${trainset}_noseg data/${trainset}_e2e_hires
utils/fix_data_dir.sh data/${trainset}_e2e_hires

mfccdir=mfcc_hires
steps/make_mfcc.sh --nj 50 --mfcc-config conf/mfcc_hires.conf --cmd "$train_cmd" data/${trainset}_e2e_hires exp/make_hires/${trainset}_e2e_hires $mfccdir
steps/compute_cmvn_stats.sh data/${trainset}_e2e_hires exp/make_hires/${trainset}_e2e_hires $mfccdir

cp -r data/${trainset}_e2e_hires data/${trainset}_e2e_hires_spk
mv data/${trainset}_e2e_hires_spk/feats.scp data/${trainset}_e2e_hires_spk/feats_ori.scp
feat-to-len scp:data/${trainset}_e2e_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/${trainset}_e2e_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/${trainset}_e2e_hires_spk/utt2spk data/${trainset}_e2e_hires_spk/align1.pdf data/${trainset}_e2e_hires_spk/num_spk > data/${trainset}_e2e_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/${trainset}_e2e_hires_spk/spk > data/${trainset}_e2e_hires_spk/spk.ark
analyze-counts --binary=false ark:data/${trainset}_e2e_hires_spk/spk data/${trainset}_e2e_hires_spk/spk_count

paste-feats scp:data/${trainset}_e2e_hires_spk/feats_ori.scp ark:data/${trainset}_e2e_hires_spk/spk.ark ark,scp:data/${trainset}_e2e_hires_spk/feats.ark,data/${trainset}_e2e_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/${trainset}_e2e_hires_spk



bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01




bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1






bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_b6n6" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_b6n6_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_b6n6_adaptlayer7_actSig_epoch7_lr10.01_lr20.01




bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_b6n6" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_b6n6_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_b6n6_adaptlayer7_actSig_epoch7_lr10.01_lr20.01




#########################################

# cut len

bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454
feat-to-len scp:data/eval2000_hires_spk/feats.scp ark,t:- | awk '{print $2}' | sort -u | python3 local/chain/adaptation/generate_allowed_lenghts.py > exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --allowed-lengths-file exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01




bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454
feat-to-len scp:data/rt03_hires_spk/feats.scp ark,t:- | awk '{print $2}' | sort -u | python3 local/chain/adaptation/generate_allowed_lenghts.py > exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --allowed-lengths-file exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt \
 rt03_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01






bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 --allowed-lengths-file exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1






bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 --allowed-lengths-file exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt \
 rt03_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1







bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus_mmionly" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0 --mmi-scale 1 \
 --allowed-lengths-file exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0/allowed_lengths.txt \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_mmionly_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_rnnlmplus_mmionly_adaptlayer7_actSig_epoch7_lr10.01_lr20.01








