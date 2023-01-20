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

bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_oracle_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/fst_oracle_eval2000/1BEST_fst \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true --nbest_num 200 --cross_utt true --seq_len 100 --reset_history true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 --decode_dir_suffix pytorch_crossutt100_lstm_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --tied false \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --use_gpu true --use-nbest true --nbest_num 200 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 --decode-dir-suffix-forward pytorch_crossutt100_lstm_dim2048_drop15 --decode-dir-suffix-backward pytorch_crossutt100_lstm_back_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --tied false \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_bpe_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1 data/lang_bpe_sw1_fsh_fg \
 eval2000_hires_spk decode_eval2000_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8



bash local/chain/adaptation/LHUC/LHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_oracle_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/fst_oracle_rt03/1BEST_fst \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true --nbest_num 200 --cross_utt true --seq_len 100 --reset_history true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 --decode_dir_suffix pytorch_crossutt100_lstm_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --tied false \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --use_gpu true --use-nbest true --nbest_num 200 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 --decode-dir-suffix-forward pytorch_crossutt100_lstm_dim2048_drop15 --decode-dir-suffix-backward pytorch_crossutt100_lstm_back_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --tied false \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_bpe_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1 data/lang_bpe_sw1_fsh_fg \
 rt03_hires_spk decode_rt03_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8



## BLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_oracle_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/fst_oracle_eval2000/1BEST_fst \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done




bash local/chain/adaptation/LHUC/BLHUC_adaptation_e2e.sh \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --LM-path data/lang_bpe_ --pre_out_layer prefinal-l --pre_out_dim 256 \
 --xent_regularize 0.1 --mmi_scale 1.0 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_oracle_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi/fst_oracle_rt03/1BEST_fst \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_oracle_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done


