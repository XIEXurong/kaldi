
# baseline

bash local/chain/oldscripts/decode.sh

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 "eval2000_fbk_40 rt03_fbk_40" exp/chain/old_models/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp



# BLHUC adapt

num_chunk=64
tag=_samllstream


bash local/chain/oldscripts/1BEST_cnn_tdnn/BLHUC_iv/decode_eval2000.sh --frames_per_eg "150,100,50" --tag "$tag" --num-chunk ${num_chunk} --baseline cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp --layer-num 12 --epoch-num 7 --lr1 0.01 --lr2 0.01 "" 0.0001

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 eval2000_fbk_40_spk exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01

bash local/rnnlm/run_rescore_new.sh --use-nbest true --nbest-num 100 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 eval2000_fbk_40_spk exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01



bash local/pytorchnn/run_nnlm_decode_mod.sh \
 --LM-path data/old_models/lang_ \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000_fbk_40_spk" exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01



bash local/chain/oldscripts/1BEST_cnn_tdnn/BLHUC_iv/decode_rt03.sh --frames_per_eg "150,100,50" --tag "$tag" --num-chunk ${num_chunk} --baseline cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp --layer-num 12 --epoch-num 7 --lr1 0.01 --lr2 0.01 "" 0.0001

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 rt03_fbk_40_spk exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rt03_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01

bash local/rnnlm/run_rescore_new.sh --use-nbest true --nbest-num 100 \
 --LM-path data/old_models/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 rt03_fbk_40_spk exp/chain/old_models/1BEST_cnn_tdnn/BLHUC_iv/cnn_tdnn_fbk_40_iv_1a_ep6_multijobs_specmaskonline4_sp_BLHUC_12Layergap7_new_rt03_rnnlmplus_KL0.0001_increase${tag}_batch${num_chunk}_epoch7_lr10.01_lr20.01







