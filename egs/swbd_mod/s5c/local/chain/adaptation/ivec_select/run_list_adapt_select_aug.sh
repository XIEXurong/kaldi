cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c


# bash readme_eval2000_select_data_fram_train.sh


# LHUC adapt

N=_sub10
aug_num=1
tag=_perspk_cos
w1=1
w2=0.1
lab_dir=exp/chain/cnn_tdnn1a_sp_decode_aug/decode_eval2000_hires_spk${N}_sw1_fsh_fg_1best_aug${aug_num}${tag}

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline cnn_tdnn1a_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000${N}_aug${aug_num}${tag} \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000${N}_aug${aug_num}${tag}_weight${w1}_${w2}" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 0.0 \
 --frames-per-eg "150,100,50" \
 --deriv-weights-scp $lab_dir/weight${w1}_${w2}.scp \
 eval2000_hires_spk${N}_aug${aug_num}${tag} \
 $lab_dir eval2000_hires_spk





