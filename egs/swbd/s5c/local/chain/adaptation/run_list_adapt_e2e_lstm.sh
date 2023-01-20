cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnn_lstm_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01





bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_lstm_1a_iv_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01


