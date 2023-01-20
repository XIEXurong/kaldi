cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnnf_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_eval2000_mmice_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 rt03_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_rt03_mmice_adaptlayer7_actSig_batch64_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

