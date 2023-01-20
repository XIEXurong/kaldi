cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done




bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5

bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5




## BLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_pca3_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_pca3_eval2000 \
 --LM-path data/lang_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmpluspytfcrossutt" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_sw1_fsh_fg \
 eval2000_hires_spk decode_eval2000_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6




bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_pca3_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_pca3_rt03 \
 --LM-path data/lang_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rnnlmpluspytfcrossutt" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_ivpca3_bi_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_sw1_fsh_fg \
 rt03_hires_spk decode_rt03_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

