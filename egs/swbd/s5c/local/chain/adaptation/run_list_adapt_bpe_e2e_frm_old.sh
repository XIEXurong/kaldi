cd /home/hei/works/kaldi_bayes_adapt/egs/swbd_mod/s5c

dir=exp/chain/adaptation/LHUC_e2e
for f1 in `ls $dir/ | grep "e2e_cnn_tdnnf"`; do
    for f2 in `ls $dir/$f1 | grep "decode_"`; do
        rm -r $dir/$f1/$f2/score_*
        rm -r $dir/$f1/$f2/scoring
    done
    rm -r $dir/$f1/{0.raw,0.mdl,20.mdl,21.mdl,41.mdl,42.mdl,egs,cache*,configs/ref.raw}
done




bash local/chain/adaptation/generate_1best_lat_all.sh --lm_scale 8 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe_mmice_specaugkaldi/decode_f3m/decode_eval2000_sw1_fsh_fg

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh --stop_stage 1 --adapt_model exp/chain/e2e_cnn_tdnnf_1a_iv_bpe_mmice_specaugkaldi/decode_f3m/final.mdl \
 --baseline e2e_cnn_tdnnf_1a_iv_bpe_mmice_specaugkaldi \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 --frames-per-eg "150,100,50" \
 --adapt-base adaptation/LHUC_e2e \
 eval2000_hires_spk \
 exp/chain/e2e_cnn_tdnnf_1a_iv_bpe_mmice_specaugkaldi/decode_f3m/decode_eval2000_sw1_fsh_fg/1BEST_lat/score_8_0.0 \
 eval2000_hires_spk


. ./path.sh
. ./cmd.sh
dir=exp/chain/adaptation/LHUC_e2e/e2e_cnn_tdnnf_1a_iv_bpe_mmice_specaugkaldi_BLHUC_eval2000_adaptlayer7_actSig_batch64_epoch7_lr10.01_lr20.01
utils/mkgraph.sh --self-loop-scale 1.0 data/lang_bpe_sw1_tg $dir $dir/graph_sw1_tg

decode_nj=50
graph_dir=$dir/graph_sw1_tg
decode_set=eval2000
steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
  --nj $decode_nj --cmd "$decode_cmd" \
  --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
  $graph_dir data/${decode_set}_hires_spk \
  $dir/decode_${decode_set}_sw1_tg
steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
data/lang_bpe_sw1_{tg,fsh_fg} data/${decode_set}_hires \
$dir/decode_${decode_set}_sw1_{tg,fsh_fg}

for decode_set in eval2000 rt03; do
    for lm in tg fsh_fg; do
        dir1=$dir/decode_${decode_set}_sw1_${lm}
        is_rt03="" && if [[ "$decode_set" == "rt03" ]]; then is_rt03=_rt03; fi
        bash compute_score.sh $dir1 > $dir1/../scoring_all$is_rt03
    done
done

