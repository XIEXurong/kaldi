cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnnf_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



## LHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 0.0 \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01





bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 0.0 \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 rt03_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_rt03_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_rt03_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01




## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01





bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 rt03_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_rt03_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_rt03_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01




