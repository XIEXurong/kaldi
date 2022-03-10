cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

# dir=cnn_tdnn1a_sp
# for f in `ls exp/chain/$dir/ | grep "decode_"`; do
    # rm -r exp/chain/$dir/$f/score_*
# done


# dir=cnn_tdnn1a_sp
# for test in eval2000 rt03; do
    # for f in `ls exp/chain/$dir/ | grep "decode_" | grep "$test" | grep -E "rnnlm|lstm|transformer"`; do
        # cat exp/chain/$dir/$f/scoring_all | grep "ctm.filt.sys"
    # done | sort > exp/chain/$dir/scoring_${test}_nnlm_sort
# done


# tdnn

bash local/chain/run_tdnn.sh

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir=exp/chain/tdnn7r_sp/decode_${decode_set}_sw1_${lm}
        if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir >> $dir/../scoring_all$is_rt03
    done
done


bash local/chain/run_cnn_tdnn_mod.sh

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir=exp/chain/cnn_tdnn1a_sp/decode_${decode_set}_sw1_${lm}
        if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir >> $dir/../scoring_all$is_rt03
    done
done




# lstm lm

## 2 direction lstm

bash local/rnnlm/run_tdnn_lstm.sh --run-lat-rescore false --run-nbest-rescore false --dir exp/rnnlm/rnnlm_lstm_1e
bash local/rnnlm/run_tdnn_lstm_back.sh --run-lat-rescore false --dir exp/rnnlm/rnnlm_lstm_1e_backward

bash local/rnnlm/run_rescore_new.sh \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 4 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


bash local/rnnlm/run_rescore_new.sh --use-nbest true \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/rnnlm/run_rescore_new.sh --use-nbest true --nbest-weight 0.9 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/rnnlm/run_rescore_new_2weights_backonly.sh \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


bash local/rnnlm/run_rescore_new_2weights.sh \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward \
 --decode-dir-suffix-forward rnnlm_1e \
 --decode-dir-suffix-backward rnnlm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


bash local/rnnlm/run_rescore_new.sh --use-nbest true --nbest-num 20 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/old_models/rnnlm_lstm_1e_large_drop_e40 \
 --dir-backward exp/rnnlm/old_models/rnnlm_lstm_1e_backward_large_drop_e40 \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40 \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


## 2 direction lstm large (old)

bash local/rnnlm/rnnlm_copy_map.sh --new-word-list exp/rnnlm/rnnlm_lstm_1e/config/words.txt exp/rnnlm/old_models/rnnlm_lstm_1e_large_drop_e40 exp/rnnlm/rnnlm_lstm_1e_large_drop_e40_map
bash local/rnnlm/rnnlm_copy_map.sh --new-word-list exp/rnnlm/rnnlm_lstm_1e_backward/config/words.txt exp/rnnlm/old_models/rnnlm_lstm_1e_backward_large_drop_e40 exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40_map

bash local/rnnlm/run_rescore_new.sh --lattice-prune-beam 6 --ngram_order 6 \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_lstm_1e_large_drop_e40_map \
 --dir-backward exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40_map \
 --decode-dir-suffix-forward rnnlm_1e_large_drop_e40_map \
 --decode-dir-suffix-backward rnnlm_1e_back_large_drop_e40_map \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


## 2 direction lstm large

bash local/rnnlm/run_tdnn_lstm_mod_large_drop_e40.sh --stage 1 --run-lat-rescore false --run-nbest-rescore false --dir exp/rnnlm/rnnlm_lstm_1e_large_drop_e40
bash local/rnnlm/run_tdnn_lstm_back_mod_large_drop_e40.sh --stage 1 --run-lat-rescore false --dir exp/rnnlm/rnnlm_lstm_1e_backward_large_drop_e40




### nbest LM score recombine

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_rnnlm_20best_interp_lstm_0.5
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_20best_0.8"
lm_scale_list="0.5 0.5"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_rnnlm_20best_interp_2lstm_0.5
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_20best_0.8"
lm_scale_list="0.5 0.25 0.25"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_rt03_sw1_fsh_fg_rnnlm_20best_interp_lstm_0.5
dataset=data/rt03
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_rt03_sw1_fsh_fg_rnnlm_1e_20best_0.8"
lm_scale_list="0.5 0.5"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_rt03_sw1_fsh_fg_rnnlm_20best_interp_2lstm_0.5
dataset=data/rt03
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_rt03_sw1_fsh_fg_rnnlm_1e_20best_0.8 $dir/../decode_rt03_sw1_fsh_fg_rnnlm_1e_back_20best_0.8"
lm_scale_list="0.5 0.25 0.25"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base



## blstm

bash local/rnnlm/run_tdnn_blstm.sh --run-lat-rescore false --run-nbest-rescore false

bash local/rnnlm/run_rescore_new.sh \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_blstm_1e/forward \
 --dir-backward exp/rnnlm/rnnlm_blstm_1e/backward \
 --decode-dir-suffix-forward rnnlm_blstm_1e \
 --decode-dir-suffix-backward rnnlm_blstm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/rnnlm/run_rescore_new.sh --use-nbest true \
 --LM-path data/lang_ --LM sw1_fsh_fg --data-dir data \
 --dir-forward exp/rnnlm/rnnlm_blstm_1e/forward \
 --dir-backward exp/rnnlm/rnnlm_blstm_1e/backward \
 --decode-dir-suffix-forward rnnlm_blstm_1e \
 --decode-dir-suffix-backward rnnlm_blstm_1e_back \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


### nbest LM score recombine

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_rnnlm_20best_interp_blstm_0.5
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_back_20best_0.8"
lm_scale_list="0.5 0.25 0.25"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_rt03_sw1_fsh_fg_rnnlm_20best_interp_blstm_0.5
dataset=data/rt03
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_rt03_sw1_fsh_fg_rnnlm_blstm_1e_20best_0.8 $dir/../decode_rt03_sw1_fsh_fg_rnnlm_blstm_1e_back_20best_0.8"
lm_scale_list="0.5 0.25 0.25"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_rnnlm_20best_interp_lstm+blstm_0.35+0.35+0.3
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_back_20best_0.8"
lm_scale_list="0.35 0.35 0.15 0.15"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_rt03_sw1_fsh_fg_rnnlm_20best_interp_lstm+blstm_0.35+0.35+0.3
dataset=data/rt03
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_rt03_sw1_fsh_fg_rnnlm_1e_20best_0.8 $dir/../decode_rt03_sw1_fsh_fg_rnnlm_blstm_1e_20best_0.8 $dir/../decode_rt03_sw1_fsh_fg_rnnlm_blstm_1e_back_20best_0.8"
lm_scale_list="0.35 0.35 0.15 0.15"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_rnnlm_20best_interp_2lstm+blstm_0.35+0.35+0.3
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_1e_back_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_rnnlm_blstm_1e_back_20best_0.8"
lm_scale_list="0.35 0.175 0.175 0.15 0.15"
bash local/rnnlm/nbest_scoring_interpolation.sh --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base



# pytorchnn lm

# lstm lm
bash local/pytorchnn/run_nnlm_train_mod.sh \
 --pytorch-path exp/pytorchnn_lm/pytorch_lstm --model-type LSTM \
 --embedding-dim 650 --hidden-dim 650 --nlayers 2 --learning-rate 5

# nbest with weight=0.8
bash local/pytorchnn/run_nnlm_decode_mod.sh --use-nbest true \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

# lattice with weight=0.8 and beam=5
bash local/pytorchnn/run_nnlm_decode_mod.sh \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_mod.sh --weight 0.4 \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


# transformer lm
bash local/pytorchnn/run_nnlm_train_mod.sh --stage 1

bash local/pytorchnn/run_nnlm_decode_mod.sh --use-nbest true \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_mod.sh \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_mod.sh --weight 0.4 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp




bash local/pytorchnn/run_nnlm_train_back_mod.sh \
 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back --model-type LSTM \
 --embedding-dim 650 --hidden-dim 650 --nlayers 2 --learning-rate 5

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --use-nbest true \
 --decode-dir-suffix-forward pytorch_lstm --decode-dir-suffix-backward pytorch_lstm_back --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --weight 0.4 \
 --decode-dir-suffix-forward pytorch_lstm --decode-dir-suffix-backward pytorch_lstm_back --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp



bash local/pytorchnn/run_nnlm_train_back_mod.sh --stage 1 # transformer

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --use-nbest true \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --weight 0.4 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp



### nbest LM score recombine

dir=exp/chain/cnn_tdnn1a_sp/decode_eval2000_sw1_fsh_fg_pytorch_lstm_transformer_20best_interp_0.2+0.4+0.4
dataset=data/eval2000
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_eval2000_sw1_fsh_fg_pytorch_lstm_20best_0.8 $dir/../decode_eval2000_sw1_fsh_fg_pytorch_transformer_20best_0.8"
lm_scale_list="0.2 0.4 0.4"
bash local/rnnlm/nbest_scoring_interpolation.sh --nnlm-score-file-name lmwt.nn.sum \
 --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base


dir=exp/chain/cnn_tdnn1a_sp/decode_rt03_sw1_fsh_fg_pytorch_lstm_transformer_20best_interp_0.2+0.4+0.4
dataset=data/rt03
LM_base=data/lang_sw1_fsh_fg
LM_list="$dir/../decode_rt03_sw1_fsh_fg_pytorch_lstm_20best_0.8 $dir/../decode_rt03_sw1_fsh_fg_pytorch_transformer_20best_0.8"
lm_scale_list="0.2 0.4 0.4"
bash local/rnnlm/nbest_scoring_interpolation.sh --nnlm-score-file-name lmwt.nn.sum \
 --scoring-opt "--min-lmwt 9 --max-lmwt 12" --lm-scale-list "$lm_scale_list" $dir "$LM_list" $dataset $LM_base



# pytorchnn lm large

bash local/pytorchnn/run_nnlm_train_mod.sh --stage 1 \
 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim1024_nodrop --model-type LSTM \
 --embedding-dim 1024 --hidden-dim 1024 --nlayers 2 --learning-rate 5 --dropout 0

bash local/pytorchnn/run_nnlm_decode_mod.sh --use-nbest true \
 --decode-dir-suffix pytorch_lstm_dim1024_nodrop --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim1024_nodrop \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 1024 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


bash local/pytorchnn/run_nnlm_train_mod_new.sh --stage 1 \
 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim2048_drop15 --model-type LSTM \
 --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --learning-rate 5 --dropout 0.15 --tied false

bash local/pytorchnn/run_nnlm_decode_mod.sh --use-nbest true \
 --decode-dir-suffix pytorch_lstm_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" --tied false \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp


bash local/pytorchnn/run_nnlm_train_back_mod_new.sh --stage 1 \
 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 --model-type LSTM \
 --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 --learning-rate 5 --dropout 0.15 --tied false

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --use-nbest true \
 --decode-dir-suffix-forward pytorch_lstm_dim2048_drop15 --decode-dir-suffix-backward pytorch_lstm_back_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" --tied false \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp



bash local/pytorchnn/run_nnlm_decode_mod.sh --beam 6 \
 --decode-dir-suffix pytorch_lstm_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" --tied false \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --beam 6 \
 --decode-dir-suffix-forward pytorch_lstm_dim2048_drop15 --decode-dir-suffix-backward pytorch_lstm_back_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" --tied false \
 "eval2000" exp/chain/cnn_tdnn1a_sp

bash local/pytorchnn/run_nnlm_decode_back_mod.sh --beam 7 --beam-forward 6 \
 --decode-dir-suffix-forward pytorch_lstm_dim2048_drop15 --decode-dir-suffix-backward pytorch_lstm_back_dim2048_drop15 --pytorch-path exp/pytorchnn_lm/pytorch_lstm_back_dim2048_drop15 \
 --model-type LSTM --embedding-dim 1024 --hidden-dim 2048 --nlayers 2 \
 --scoring-opts "--min-lmwt 9 --max-lmwt 12" --tied false \
 "rt03" exp/chain/cnn_tdnn1a_sp




