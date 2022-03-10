cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

bash local/chain/adaptation/decode_all_cnn_ivector_subN.sh --LM fsh_fg --data-dir data \
--exp-dir exp --ext _hires eval2000 \
exp/chain/cnn_tdnn1a_sp_subN exp/chain/cnn_tdnn1a_sp

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/cnn_tdnn1a_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg
done



# LHUC adapt

N=_sub10

if [ ! -d exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg ]; then
  ln -s $PWD/exp/chain/cnn_tdnn1a_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg
fi

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline cnn_tdnn1a_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 0.0 \
 --frames-per-eg "150,100,50" \
 eval2000_hires_spk$N \
 exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC/cnn_tdnn1a_sp_LHUC_eval2000${N}_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done


# BLHUC adapt

N=_sub10

if [ ! -d exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg ]; then
  ln -s $PWD/exp/chain/cnn_tdnn1a_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg
fi

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline cnn_tdnn1a_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 eval2000_hires_spk$N \
 exp/chain/cnn_tdnn1a_sp/decode_eval2000_hires${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


for decode_set in eval2000_hires_spk; do
    for lm in tg fsh_fg; do
        dir=exp/chain/adaptation/LHUC/cnn_tdnn1a_sp_BLHUC_eval2000${N}_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01/decode_${decode_set}_sw1_${lm}
        bash compute_score.sh $dir >> $dir/../scoring_all
    done
done











