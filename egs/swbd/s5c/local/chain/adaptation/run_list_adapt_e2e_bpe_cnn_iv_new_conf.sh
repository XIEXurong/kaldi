cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done




bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg
bash local/chain/adaptation/generate_1best_lat_all_weights.sh \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/final.mdl 1BEST_lat/score_10_0.0
cat exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.*.ark | \
 awk '{sum=0;for(i=3;i<NF;i++)sum+=$i; print sum/(NF-3),$1}' | sort -r | awk '{print $2,$1}' > exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort
utt_num=`cat exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.scp | wc -l | awk '{print int($1*0.8)}'`
head -n $utt_num exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort > exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort0.8


bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg
bash local/chain/adaptation/generate_1best_lat_all_weights.sh \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/final.mdl 1BEST_lat/score_10_0.0
cat exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.*.ark | \
 awk '{sum=0;for(i=3;i<NF;i++)sum+=$i; print sum/(NF-3),$1}' | sort -r | awk '{print $2,$1}' > exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort
utt_num=`cat exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.scp | wc -l | awk '{print int($1*0.8)}'`
head -n $utt_num exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort > exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort0.8


for decode_set in eval2000 rt03; do
    lab_dir=exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_${decode_set}_sw1_fsh_fg/1BEST_fst/score_10_0.0
    mkdir -p ${lab_dir}_best0.8
    cat ${lab_dir}/fst.*.scp | perl local/chain/adaptation/find_pdf_stdin.pl exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_${decode_set}_sw1_fsh_fg/1BEST_weights/score_10_0.0/weights.sort0.8 | sort > ${lab_dir}_best0.8/fst.1.scp
    echo "1" > ${lab_dir}_best0.8/num_jobs
done




## LHUC adapt

# 1best

bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_best0.8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0_best0.8 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_best0.8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_best0.8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0_best0.8 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_best0.8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




## BLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_best0.8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0_best0.8 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_best0.8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_best0.8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0_best0.8 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_best0.8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done


