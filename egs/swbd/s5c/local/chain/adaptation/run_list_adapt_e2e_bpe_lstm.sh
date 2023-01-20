cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnn_blstm_1a"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,41.mdl,egs,cache*,configs/ref.raw}
done



## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg

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
 --act "Sig" --tag "_eval2000_e2ehires" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk

for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done



bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



bash local/chain/adaptation/generate_1best_fst_all.sh --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01


lang=data/lang_bpe_sw1_fsh_fg
N=20
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --cuda_id 1 --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod_crossutt.sh --use_gpu true --cuda_id 1 --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_crossutt100_transformer --weight $weight --nbest_num $N --seq_len 100 --reset_history true \
     "${decode_set}" $baseline
done



lang=data/lang_bpe_sw1_fsh_fg
N=100
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --cuda_id 0 --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod_crossutt.sh --use_gpu true --cuda_id 0 --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_crossutt100_transformer --weight $weight --nbest_num $N --seq_len 100 --reset_history true \
     "${decode_set}" $baseline
done



bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 --conv_sort_list exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/both_text.sorted.txt \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_conv_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_bpe_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_bpe_sw1_fsh_fg \
 eval2000_hires_spk decode_eval2000_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 100 --conv_sort_list exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/both_text.sorted.txt \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt100_conv_transformer --py_nnlm exp/pytorchnn_lm/pytorch_transformer --LM_path data/lang_bpe_ \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_bpe_sw1_fsh_fg \
 rt03_hires_spk decode_rt03_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6


bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 128 --conv_sort_list exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/both_text.sorted.txt \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt128_conv_transformer24L --py_nnlm exp/pytorchnn_lm/pytorch_transformer24L --LM_path data/lang_bpe_ \
 --pynnlm_config "--model-type Transformer --embedding-dim 512 --hidden-dim 512 --nlayers 24 --nhead 8" \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_bpe_sw1_fsh_fg \
 eval2000_hires_spk decode_eval2000_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

bash local/chain/nnlmlat_pynnlm_rescore_crossutt.sh --use_gpu true --seq_len 128 --conv_sort_list exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/both_text.sorted.txt \
 --N 20 --weight 0.5 --py_nnlm_tag pytorch_crossutt128_conv_transformer24L --py_nnlm exp/pytorchnn_lm/pytorch_transformer24L --LM_path data/lang_bpe_ \
 --pynnlm_config "--model-type Transformer --embedding-dim 512 --hidden-dim 512 --nlayers 24 --nhead 8" \
 exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_adaptlayer7_actSig_epoch7_lr10.01_lr20.01 data/lang_bpe_sw1_fsh_fg \
 rt03_hires_spk decode_rt03_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6





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
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.025 --mmi-scale 1 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1




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
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_mmi" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.025 --mmi-scale 1 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1



# cpu

lang=data/lang_bpe_sw1_fsh_fg
N=20
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_cpu_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_cpu_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_cpu_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu false --use-nbest true --other_opt "--stage 5" --LM_path data/lang_bpe_ \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_cpu_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_cpu_crossutt100_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_cpu_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_cpu_crossutt100_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmi_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod_crossutt.sh --use_gpu false --use-nbest true --other_opt "--stage 5" --LM_path data/lang_bpe_ \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_cpu_crossutt100_transformer --weight $weight --nbest_num $N --seq_len 100 --reset_history true \
     "${decode_set}" $baseline
done




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
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1




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
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_mmice" \
 --epoch-num 7 --lr1 0.1 --lr2 0.1 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --xent-regularize 0.1 --mmi-scale 1 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_10_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1



lang=data/lang_bpe_sw1_fsh_fg
N=20
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true --LM_path data/lang_bpe_ --other_opt '--gpu_wait true --stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_crossutt100_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_mmice_adaptlayer7_actSig_epoch7_lr10.1_lr20.1
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod_crossutt.sh --use_gpu true --use-nbest true --LM_path data/lang_bpe_ --other_opt '--gpu_wait true --stage 5 --limit_num_gpus_cmd "\"\""' \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_crossutt100_transformer --weight $weight --nbest_num $N --seq_len 100 --reset_history true \
     "${decode_set}" $baseline
done



###############################################################################################





## LHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_fst_all.sh --lm_scale 8 --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_s8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_8_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_eval2000_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01





bash local/chain/adaptation/generate_1best_fst_all.sh --lm_scale 8 --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_s8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-init 0.0 \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_8_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_rt03_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



lang=data/lang_bpe_sw1_fsh_fg
N=20
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_${decode_set}_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_LHUC_e2e_${decode_set}_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' --gpu_nj 50 \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done




## BLHUC adapt

# 1best


bash local/chain/adaptation/generate_1best_fst_all.sh --lm_scale 8 --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_eval2000_e2ehires_rnnlmplus_s8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_8_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_eval2000_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01





bash local/chain/adaptation/generate_1best_fst_all.sh --lm_scale 8 --lang data/lang_e2e_bpe exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454

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
 --act "Sig" --tag "_rt03_e2ehires_rnnlmplus_s8" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 rt03_e2e_hires_spk \
 exp/chain/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.454/1BEST_fst/score_8_0.0 \
 rt03_hires_spk


for decode_set in rt03_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_bpe_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "rt03_hires_spk" exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_rt03_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01



lang=data/lang_bpe_sw1_fsh_fg
N=20
weight=0.5

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/nbest_rescore.sh --N $N --skip_scoring true $lang data/${decode_set}_hires_spk $dir ${dir}_${N}best

    cp -r ${dir}_${N}best ${dir}_0graph_pytorch_transformer_${N}best_${weight}
    nj=`cat ${dir}_${N}best/num_jobs`
    for i in `seq 1 $nj`; do
        cat ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/acwt | \
          awk '{print $1,0}' > ${dir}_0graph_pytorch_transformer_${N}best_${weight}/archives.$i/lmwt.nolm
    done
done

for decode_set in eval2000 rt03; do
    baseline=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_BLHUC_e2e_${decode_set}_e2ehires_rnnlmplus_s8_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    dir=$baseline/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6
    bash local/pytorchnn/run_nnlm_decode_mod.sh --use_gpu true --use-nbest true --LM_path data/lang_bpe_ --other_opt '--stage 5 --limit_num_gpus_cmd "\"\""' --gpu_nj 50 \
     --decode_dir $dir --decode_dir_suffix 0graph_pytorch_transformer --weight $weight --nbest_num $N \
     "${decode_set}" $baseline
done





