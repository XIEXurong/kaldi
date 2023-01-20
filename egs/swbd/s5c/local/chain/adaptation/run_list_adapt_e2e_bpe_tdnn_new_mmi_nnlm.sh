cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



##################################################




## LHUC adapt

# 1best

bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8

bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bpe3g_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --LM-path data/lang_bpe_ --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_eval2000_e2ehires_mmice_transformer" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_LHUC_e2e_eval2000_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_LHUC_e2e_eval2000_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1



bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8

bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bpe3g_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --LM-path data/lang_bpe_ --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_rt03_e2ehires_mmice_transformer" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_LHUC_e2e_rt03_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_LHUC_e2e_rt03_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1



## BLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bpe3g_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --LM-path data/lang_bpe_ --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_eval2000_e2ehires_mmice_transformer" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_BLHUC_e2e_eval2000_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_BLHUC_e2e_eval2000_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1



bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bpe3g_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --LM-path data/lang_bpe_ --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_rt03_e2ehires_mmice_transformer" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bpe3g_mmice/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_BLHUC_e2e_rt03_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bpe3g_mmice_BLHUC_e2e_rt03_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1


