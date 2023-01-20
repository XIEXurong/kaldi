cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done


## LHUC adapt

# 1best


bash local/chain/adaptation/LHUC/KLLHUC_adaptation_cnn_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --rho 0.125 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_rho0.125" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_KLLHUC_e2e_eval2000_e2ehires_rho0.125_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/LHUC/KLLHUC_adaptation_cnn_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --rho 0.125 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_rho0.125" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_KLLHUC_e2e_rt03_e2ehires_rho0.125_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



##################################################




## MAPLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/MAPLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.00000001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_MAPLHUC_e2e_eval2000_e2ehires_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHUC/MAPLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.00000001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi_MAPLHUC_e2e_rt03_e2ehires_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

