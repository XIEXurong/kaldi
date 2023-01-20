cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done

dir=exp/chain/adaptation/HUB_e2e
for f1 in `ls $dir/ | grep "e2e_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done

dir=exp/chain/adaptation/PAct_e2e
for f1 in `ls $dir/ | grep "e2e_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done

dir=exp/chain/adaptation/LHN_e2e
for f1 in `ls $dir/ | grep "e2e_tdnn"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,21.mdl,42.mdl,egs,cache*,configs/ref.raw}
done




##################################################




## LHUC adapt

# 1best


bash local/chain/adaptation/LHUC/KLLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 --rho 0.125 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_eval2000_e2ehires_mmice_rho0.125" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bi_mmice_KLLHUC_e2e_eval2000_e2ehires_mmice_rho0.125_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/LHUC/KLLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0  --rho 0.125 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_rt03_e2ehires_mmice_rho0.125" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bi_mmice_KLLHUC_e2e_rt03_e2ehires_mmice_rho0.125_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



##################################################




## MAPLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/MAPLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.00000001 0.0000001 0.000001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bi_mmice_MAPLHUC_e2e_eval2000_e2ehires_mmice_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHUC/MAPLHUC_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.00000001 0.0000001 0.000001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001 0.00001" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Sig" --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bi_mmice_MAPLHUC_e2e_rt03_e2ehires_mmice_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



##################################################




## HUB adapt

# 1best

bash local/chain/adaptation/HUB/HUB_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Idnt" --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.00001 --lr2 0.00001 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/HUB_e2e/e2e_tdnnf_7r_bi_mmice_HUB_e2e_eval2000_e2ehires_mmice_adaptlayer14_actIdnt_epoch7_lr10.00001_lr20.00001/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/HUB/HUB_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Idnt" --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.00001 --lr2 0.00001 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/HUB_e2e/e2e_tdnnf_7r_bi_mmice_HUB_e2e_rt03_e2ehires_mmice_adaptlayer14_actIdnt_epoch7_lr10.00001_lr20.00001/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



## BHUB adapt

# 1best

bash local/chain/adaptation/HUB/BHUB_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Idnt" --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.00001 --lr2 0.00001 --param-mean-init 0.0 --param-std-init 0.01 \
 --prior-mean "0.0 0.0" --prior-std "0.01 0.01" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/HUB_e2e/e2e_tdnnf_7r_bi_mmice_BHUB_e2e_eval2000_e2ehires_mmice_adaptlayer14_actIdnt_epoch7_lr10.00001_lr20.00001/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/HUB/BHUB_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --act "Idnt" --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.00001 --lr2 0.00001 --param-mean-init 0.0 --param-std-init 0.01 \
 --prior-mean "0.0 0.0" --prior-std "0.01 0.01" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/HUB_e2e/e2e_tdnnf_7r_bi_mmice_BHUB_e2e_rt03_e2ehires_mmice_adaptlayer14_actIdnt_epoch7_lr10.00001_lr20.00001/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




##################################################




## PAct adapt

# 1best

bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg

bash local/chain/adaptation/PAct/PAct_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param_alpha_init 1.0 --param_beta_init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/PAct_e2e/e2e_tdnnf_7r_bi_mmice_PAct_e2e_eval2000_e2ehires_mmice_adaptlayer14_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg

bash local/chain/adaptation/PAct/PAct_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param_alpha_init 1.0 --param_beta_init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/PAct_e2e/e2e_tdnnf_7r_bi_mmice_PAct_e2e_rt03_e2ehires_mmice_adaptlayer14_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



## BPAct adapt

# 1best

bash local/chain/adaptation/PAct/BPAct_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param_alpha_mean_init 1.0 --param_alpha_std_init 1.0 --param_beta_mean_init 0.0 --param_beta_std_init 1.0 \
 --prior-alpha-mean "1.0 1.0" --prior-alpha-std "1.0 1.0" --prior-beta-mean "0.0 0.0" --prior-beta-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/PAct_e2e/e2e_tdnnf_7r_bi_mmice_BPAct_e2e_eval2000_e2ehires_mmice_adaptlayer14_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/PAct/BPAct_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param_alpha_mean_init 1.0 --param_alpha_std_init 1.0 --param_beta_mean_init 0.0 --param_beta_std_init 1.0 \
 --prior-alpha-mean "1.0 1.0" --prior-alpha-std "1.0 1.0" --prior-beta-mean "0.0 0.0" --prior-beta-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/PAct_e2e/e2e_tdnnf_7r_bi_mmice_BPAct_e2e_rt03_e2ehires_mmice_adaptlayer14_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




##################################################




## LHN adapt

# 1best

bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg

bash local/chain/adaptation/LHN/LHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_LHN_e2e_eval2000_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg

bash local/chain/adaptation/LHN/LHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_LHN_e2e_rt03_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



## BLHN adapt

# 1best

bash local/chain/adaptation/LHN/BLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --KL-scale "0.01" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat --log-std true --param-std-init -2.3 \
 --prior-mean-file local/chain/adaptation/LHN/idvec.mat --prior_std 0.1 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_BLHN_e2e_eval2000_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHN/BLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --KL-scale "0.01" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat --log-std true --param-std-init -2.3 \
 --prior-mean-file local/chain/adaptation/LHN/idvec.mat --prior_std 0.1 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_BLHN_e2e_rt03_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



## MAPLHN adapt

# 1best

bash local/chain/adaptation/LHN/MAPLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --KL-scale "0.000001" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 --prior-mean-file local/chain/adaptation/LHN/idvec.mat --prior_std 0.1 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_MAPLHN_e2e_eval2000_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHN/MAPLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --KL-scale "0.000001" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 --prior-mean-file local/chain/adaptation/LHN/idvec.mat --prior_std 0.1 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_MAPLHN_e2e_rt03_e2ehires_mmice_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



## KLLHN adapt

# 1best

bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg

bash local/chain/adaptation/LHN/KLLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 --rho 0.125 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_eval2000_e2ehires_mmice_rho0.125" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_KLLHN_e2e_eval2000_e2ehires_mmice_rho0.125_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/generate_1best_fst_all.sh exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg

bash local/chain/adaptation/LHN/KLLHN_adaptation_e2e.sh \
 --baseline e2e_tdnnf_7r_bi_mmice \
 --adapt-ivector-dir "" --test-ivector-dir "" \
 --xent_regularize 0.1 --mmi_scale 1.0 --rho 0.125 \
 --adapted-layer "tdnnf2" \
 --layer-dim "160" \
 --input-config "component-node name=input_2 component=input_2 input=Append(Offset(feature1,0), Sum(Offset(Scale(-1.0,input_copy1),-1), Offset(feature1,1)), Sum(Offset(feature1,-2), Offset(feature1,2), Offset(Scale(-2.0,input_copy2),0)))" \
 --tag "_rt03_e2ehires_mmice_rho0.125" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param_init_file local/chain/adaptation/LHN/idmat.mat \
 rt03_e2e_hires_spk \
 exp/chain/e2e_tdnnf_7r_bi_mmice/decode_rt03_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHN_e2e/e2e_tdnnf_7r_bi_mmice_KLLHN_e2e_rt03_e2ehires_mmice_rho0.125_adaptlayer1_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



