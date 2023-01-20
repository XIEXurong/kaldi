
for baseline in e2e_tdnnf_7r_bi_mmice e2e_tdnnf_7r_iv_bi_mmice e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bi_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep -v "transformer" | grep -v "rnnlmplus"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/scoring_all \
         data/lang_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep -v "transformer" | grep -v "rnnlmplus"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/scoring_all_rt03 \
         exp/chain/adaptation/LHUC_e2e/$dir/scoring_all \
         data/lang_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "transformer"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         data/lang_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "transformer"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         data/lang_sw1_fsh_fg
    done
done

for baseline in e2e_tdnnf_7r_bpe3g_mmice e2e_tdnnf_7r_iv_bpe3g_mmice e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep -v "transformer" | grep -v "rnnlmplus"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep -v "transformer" | grep -v "rnnlmplus"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/scoring_all_rt03 \
         exp/chain/adaptation/LHUC_e2e/$dir/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "transformer"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "transformer"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_pytorch_transformer_20best_0.8/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done
done




for baseline in e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bi_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "rnnlmpluspytfcrossutt"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "rnnlmpluspytfcrossutt"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_sw1_fsh_fg
    done
done

for baseline in e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "rnnlmpluspytfcrossutt"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "rnnlmpluspytfcrossutt"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done
done






for baseline in e2e_cnn_tdnnf_1a_iv_bi_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bi_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "rnnlmpluspytfcrossutt" | grep "sub"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "rnnlmpluspytfcrossutt" | grep "sub"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_sw1_fsh_fg
    done
done

for baseline in e2e_cnn_tdnnf_1a_iv_bpe3g_mmice_specaugkaldi e2e_cnn_tdnn_blstm_1a_iv_bpe3g_mmice_specaugkaldi; do
    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "eval2000" | grep "rnnlmpluspytfcrossutt" | grep "sub"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_eval2000_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done

    adapt_list=`ls exp/chain/adaptation/LHUC_e2e | grep ${baseline} | grep "rt03" | grep "rnnlmpluspytfcrossutt" | grep "sub"`

    for dir in $adapt_list; do
        bash local/chain/adaptation/compute_significance.sh \
         exp/chain/$baseline/decode_rt03_sw1_fsh_fg_rnnlm_1e_back_large_drop_e40_0.456_6_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         exp/chain/adaptation/LHUC_e2e/$dir/decode_rt03_hires_spk_sw1_fsh_fg_pytorch_crossutt100_lstm_back_dim2048_drop15_200best_0.8_0graph_pytorch_crossutt100_transformer_20best_0.5/scoring_all \
         data/lang_bpe_sw1_fsh_fg
    done
done


