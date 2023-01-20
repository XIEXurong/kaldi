cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnnf_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done





## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

bash local/chain/adaptation/LHUC/BLHUC_adaptation_mmi.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --xent-regularize 0 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi_BLHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi_BLHUC_eval2000_rnnlmplus_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1



# lat

bash local/chain/adaptation/LHUC/BLHUC_adaptation_mmi.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_rnnlmplus_lat" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --xent-regularize 0 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi_BLHUC_eval2000_rnnlmplus_lat_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_specaugkaldi_BLHUC_eval2000_rnnlmplus_lat_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1




