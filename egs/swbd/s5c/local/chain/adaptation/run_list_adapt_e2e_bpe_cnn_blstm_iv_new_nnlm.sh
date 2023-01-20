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

bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8

bash local/chain/adaptation/LHUC/LHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --chunk_left_context 40 --chunk_right_context 40 \
 --frames_per_chunk_primary 150 --extra_left_context 50 --extra_right_context 50 \
 --LM-path data/lang_bpe_ --pre_out_layer blstm3 --pre_out_dim 768 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_transformer" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8

bash local/chain/adaptation/LHUC/LHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --chunk_left_context 40 --chunk_right_context 40 \
 --frames_per_chunk_primary 150 --extra_left_context 50 --extra_right_context 50 \
 --LM-path data/lang_bpe_ --pre_out_layer blstm3 --pre_out_dim 768 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_transformer" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



## BLHUC adapt

# 1best

bash local/chain/adaptation/LHUC/BLHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --chunk_left_context 40 --chunk_right_context 40 \
 --frames_per_chunk_primary 150 --extra_left_context 50 --extra_right_context 50 \
 --LM-path data/lang_bpe_ --pre_out_layer blstm3 --pre_out_dim 768 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000_e2ehires_transformer" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



bash local/chain/adaptation/LHUC/BLHUC_adaptation_lstm_e2e.sh \
 --baseline e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_rt03 \
 --test-ivector-dir exp/nnet3/ivectors_rt03 \
 --chunk_left_context 40 --chunk_right_context 40 \
 --frames_per_chunk_primary 150 --extra_left_context 50 --extra_right_context 50 \
 --LM-path data/lang_bpe_ --pre_out_layer blstm3 --pre_out_dim 768 \
 --adapted-layer "cnn1 tdnn1 tdnn2 tdnn3 tdnn4 tdnn5 tdnn7" \
 --layer-dim "2560 1280 1280 1280 1280 1280 1280" \
 --KL-scale "0.0001 1 1 1 1 1 1" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_rt03_e2ehires_transformer" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/1BEST_fst/score_10_0.0 \
 rt03_hires_spk

for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --other_opt '--gpu_wait true --limit_num_gpus_cmd "\"\""' \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_transformer_adaptlayer7_actSig_epoch7_lr10.01_lr20.01

