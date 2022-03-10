cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c


# tdnn

# cnn-tdnn + kaldi version specaugment
bash local/chain/run_cnn_tdnn_mod_specaugkaldi.sh --gpu_exclusive true

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir=exp/chain/cnn_tdnn1a_specaugkaldi_sp/decode_${decode_set}_sw1_${lm}
        if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir >> $dir/../scoring_all$is_rt03
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_specaugkaldi_sp


# cnn-tdnn + kaldi version specaugment - dropout
bash local/chain/run_cnn_tdnn_mod_specaugkaldi_nodrop.sh --gpu_exclusive true

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir=exp/chain/cnn_tdnn1a_specaugkaldi_nodrop_sp/decode_${decode_set}_sw1_${lm}
        if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir >> $dir/../scoring_all$is_rt03
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_specaugkaldi_nodrop_sp


# cnn-tdnn + xie's version(4) specaugment
bash local/chain/run_cnn_tdnn_mod_specaug4.sh --gpu_exclusive true

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir=exp/chain/cnn_tdnn1a_specaug4_sp/decode_${decode_set}_sw1_${lm}
        if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir >> $dir/../scoring_all$is_rt03
    done
done

bash local/pytorchnn/run_nnlm_decode_mod.sh \
 --decode-dir-suffix pytorch_lstm --pytorch-path exp/pytorchnn_lm/pytorch_lstm \
 --model-type LSTM --embedding-dim 650 --hidden-dim 650 --nlayers 2 \
 "eval2000 rt03" exp/chain/cnn_tdnn1a_specaug4_sp

