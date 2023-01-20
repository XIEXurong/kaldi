

# SAT CNN-TDNN + SAT CNN-TDNN-BLSTM

. ./cmd.sh
. ./path.sh

N=20
for decode_set in eval2000 rt03; do
    dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithCNN_nbestalign_${decode_set}
    base_dir1=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    base_dir2=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi_SAT_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01

    lang=data/lang_bpe_sw1_fsh_fg

  bash local/chain/decode_rescore_nbest_align.sh \
    --N $N --nj 50 --beam 10 --cmd "$decode_cmd" --lm_score_file_name lmwt.interp \
    --old_acwt_weight 0.5 --new_acwt_weight 0.5 --post-decode-acwt 10.0 \
      --ivector exp/nnet3/ivectors_${decode_set} \
    $lang data/${decode_set}_hires_spk \
    $base_dir1/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5 \
    $base_dir2 $dir/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5
done



# SAT TDNN + SAT CNN-TDNN-BLSTM

. ./cmd.sh
. ./path.sh

N=20
for decode_set in eval2000 rt03; do
    dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithTDNN_nbestalign_${decode_set}
    base_dir1=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_rnnlmpluspytfcrossutt_adaptlayer7_actSig_epoch7_lr10.01_lr20.01
    base_dir2=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_iv_bpe3g_mmice_SAT_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1

    lang=data/lang_bpe_sw1_fsh_fg

  bash local/chain/decode_rescore_nbest_align.sh \
    --N $N --nj 50 --beam 10 --cmd "$decode_cmd" --lm_score_file_name lmwt.interp \
    --old_acwt_weight 0.5 --new_acwt_weight 0.5 --post-decode-acwt 10.0 \
      --ivector exp/nnet3/ivectors_${decode_set} \
    $lang data/${decode_set}_hires_spk \
    $base_dir1/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5 \
    $base_dir2 $dir/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5
done




# SAT TDNN + SAT CNN-TDNN + SAT CNN-TDNN-BLSTM

. ./cmd.sh
. ./path.sh

N=20
for decode_set in eval2000 rt03; do
    dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithCNN_decodeWithTDNN_nbestalign_${decode_set}
    base_dir1=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithCNN_nbestalign_${decode_set}
    base_dir2=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_iv_bpe3g_mmice_SAT_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1

    lang=data/lang_bpe_sw1_fsh_fg
  
  bash local/chain/decode_rescore_nbest_align.sh --stage 1 \
    --N $N --nj 50 --beam 10 --cmd "$decode_cmd" --lm_score_file_name lmwt.withlm --ac_score_file_name acwt.fusion \
    --old_acwt_weight 0.667 --new_acwt_weight 0.333 --post-decode-acwt 10.0 \
      --ivector exp/nnet3/ivectors_${decode_set} \
    $lang data/${decode_set}_hires_spk \
    $base_dir1/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5 \
    $base_dir2 $dir/decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5
done



# SAT TDNN + SAT CNN-TDNN + SAT CNN-TDNN-BLSTM

. ./cmd.sh
. ./path.sh

N=20
for decode_set in eval2000 rt03; do
    dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithCNN_decodeWithTDNN_nbestalign_${decode_set}
    base_dir1=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithCNN_nbestalign_${decode_set}
    base_dir2=exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_iv_bpe3g_mmice_SAT_LHUC_l1_BLHUC_e2e_${decode_set}_e2ehires_mmice_transformer_adaptlayer14_actSig_epoch7_lr10.1_lr20.1

    lang=data/lang_bpe_sw1_fsh_fg
    decode_dir=decode_${decode_set}_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_${N}best_0.5

  base_dir2_0=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi_SAT1_LHUC_l1_BLHUC_e2ehires_rnnlmpluspytfcrossutt_decodeWithTDNN_nbestalign_${decode_set}
  mkdir -p $dir/$decode_dir
  cp $base_dir2_0/$decode_dir/align.acwt $dir/$decode_dir/align.acwt
  
  echo "50" > $base_dir1/$decode_dir/num_jobs
  bash local/chain/decode_rescore_nbest_align.sh --stage 1 \
    --N $N --nj 50 --beam 10 --cmd "$decode_cmd" --lm_score_file_name lmwt.withlm --ac_score_file_name acwt.fusion \
    --old_acwt_weight 0.667 --new_acwt_weight 0.333 --post-decode-acwt 10.0 \
      --ivector exp/nnet3/ivectors_${decode_set} \
    $lang data/${decode_set}_hires_spk \
    $base_dir1/$decode_dir \
    $base_dir2 $dir/$decode_dir
done
