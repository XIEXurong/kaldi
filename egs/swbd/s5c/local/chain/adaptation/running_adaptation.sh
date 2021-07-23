
# bash local/chain/tuning/run_tdnn_7q.sh
# bash local/chain/tuning/run_cnn_tdnn_1a.sh

. ./path.sh

# In the paper cross entropy is used, thus label alignments are required and we need to generate 1-best lattices of the baseline decodeing results as adaptation labels.
# However, the non-1-best lattices can also be used theoretically.

bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg


# Concatenate the speaker id with the acoustic features

cp -r data/eval2000_hires data/eval2000_hires_spk
mv data/eval2000_hires_spk/feats.scp data/eval2000_hires_spk/feats_ori.scp
feat-to-len scp:data/eval2000_hires_spk/feats_ori.scp ark,t:- | perl local/chain/adaptation/gen0pdf.pl 1 > data/eval2000_hires_spk/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_hires_spk/utt2spk data/eval2000_hires_spk/align1.pdf data/eval2000_hires_spk/num_spk > data/eval2000_hires_spk/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_hires_spk/spk > data/eval2000_hires_spk/spk.ark
analyze-counts --binary=false ark:data/eval2000_hires_spk/spk data/eval2000_hires_spk/spk_count

paste-feats scp:data/eval2000_hires_spk/feats_ori.scp ark:data/eval2000_hires_spk/spk.ark ark,scp:data/eval2000_hires_spk/feats.ark,data/eval2000_hires_spk/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_hires_spk


# Create subsets as adaptation data by selecting the first N utterances from each speakers

for N in {5,10,20,40}; do

cp -r data/eval2000_hires data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/feats.scp data/eval2000_hires_spk_sub${N}/feats_ori.scp
perl local/chain/adaptation/utt2spk_split_everyN.pl data/eval2000_hires_spk/utt2spk $N | grep "_sub1$" > data/eval2000_hires_spk_sub${N}/utt2spk
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk/align1.pdf data/eval2000_hires_spk_sub${N}/utt2spk > data/eval2000_hires_spk_sub${N}/align1.pdf
perl local/chain/adaptation/segment2id.pl data/eval2000_hires_spk_sub${N}/utt2spk data/eval2000_hires_spk_sub${N}/align1.pdf data/eval2000_hires_spk_sub${N}/num_spk > data/eval2000_hires_spk_sub${N}/spk
perl local/chain/adaptation/pdf2ark_simple.pl data/eval2000_hires_spk_sub${N}/spk > data/eval2000_hires_spk_sub${N}/spk.ark
analyze-counts --binary=false ark:data/eval2000_hires_spk_sub${N}/spk data/eval2000_hires_spk_sub${N}/spk_count

paste-feats scp:data/eval2000_hires_spk_sub${N}/feats_ori.scp ark:data/eval2000_hires_spk_sub${N}/spk.ark ark,scp:data/eval2000_hires_spk_sub${N}/feats.ark,data/eval2000_hires_spk_sub${N}/feats.scp
steps/compute_cmvn_stats.sh data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/text data/eval2000_hires_spk_sub${N}/text_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk_sub${N}/text_all data/eval2000_hires_spk_sub${N}/feats.scp > data/eval2000_hires_spk_sub${N}/text
perl utils/data/get_utt2dur.sh data/eval2000_hires_spk_sub${N}
mv data/eval2000_hires_spk_sub${N}/utt2dur data/eval2000_hires_spk_sub${N}/utt2dur_all
perl local/chain/adaptation/find_pdf.pl data/eval2000_hires_spk_sub${N}/utt2dur_all data/eval2000_hires_spk_sub${N}/feats.scp > data/eval2000_hires_spk_sub${N}/utt2dur

done


# Decode the subsets and generate 1-best lattices

bash local/chain/adaptation/decode_all_ivector_subN.sh --LM fsh_fg --data-dir data \
--exp-dir exp --ext _fbk_40 eval2000 \
exp/chain/tdnn_7q_hires_sp_subN exp/chain/tdnn_7q_hires_sp

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/generate_1best_lat_all.sh exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg
done


# TDNN adaptation


# LHUC adaptation

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 1.0 \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 1.0 \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# BLHUC adaptation

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 1.0 --param-std-init 1.0 \
 --prior-mean "1.0" --prior-std "1.0" \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 1.0 --param-std-init 1.0 \
 --prior-mean "1.0" --prior-std "1.0" \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# HUB adaptation

bash local/chain/adaptation/HUB/HUB_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-init 0.0 \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/HUB/HUB_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-init 0.0 \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# BHUB adaptation

bash local/chain/adaptation/HUB/BHUB_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-mean-init 0.0 --param-std-init 0.01 \
 --prior-mean "0.0" --prior-std "1.0" \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/HUB/BHUB_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --act "Idnt" --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-mean-init 0.0 --param-std-init 0.01 \
 --prior-mean "0.0" --prior-std "1.0" \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# PAct adaptation

bash local/chain/adaptation/PAct/PAct_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-init 1.0 --param-alpha-init 0.0 \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/PAct/PAct_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-init 1.0 --param-alpha-init 0.0 \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# BPAct adaptation

bash local/chain/adaptation/PAct/BPAct_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-mean-init 1.0 --param-alpha-std-init 1.0 --param-beta-mean-init 0.0 --param-beta-std-init 1.0 \
 --prior-alpha-mean "1.0" --prior-alpha-std "1.0" --prior-beta-mean "0.0" --prior-beta-std "1.0" \
 eval2000_hires_spk \
 exp/chain/tdnn_7q_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

for N in _sub5 _sub10 _sub20 _sub40; do
bash local/chain/adaptation/PAct/BPAct_adaptation.sh \
 --baseline tdnn_7q_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "tdnn1 tdnnf2 tdnnf3 tdnnf4 tdnnf5 tdnnf6 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12 tdnnf13 tdnnf14" \
 --layer-dim "1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 0.001 0.01 0.1 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=lda component=lda input=Append(Offset(feature1, -1), feature1, Offset(feature1, 1), ReplaceIndex(ivector, t, 0))" \
 --tag "_eval2000$N" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-mean-init 1.0 --param-alpha-std-init 1.0 --param-beta-mean-init 0.0 --param-beta-std-init 1.0 \
 --prior-alpha-mean "1.0" --prior-alpha-std "1.0" --prior-beta-mean "0.0" --prior-beta-std "1.0" \
 eval2000_hires_spk$N \
 exp/chain/tdnn_7q_hires_sp_subN/decode_eval2000_hires_spk${N}_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk
done


# CNN-TDNN adaptation


# LHUC adaptation

bash local/chain/adaptation/LHUC/LHUC_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-init 0.0 \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


# BLHUC adaptation

bash local/chain/adaptation/LHUC/BLHUC_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Sig" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-mean-init 0.0 --param-std-init 1.0 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


# HUB adaptation

bash local/chain/adaptation/HUB/HUB_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-init 0.0 \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


# BHUB adaptation

bash local/chain/adaptation/HUB/BHUB_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --act "Idnt" --tag "_eval2000" \
 --epoch-num 7 --lr1 0.000001 --lr2 0.000001 --num-chunk 64 --param-mean-init 0.0 --param-std-init 0.01 \
 --prior-mean "0.0 0.0" --prior-std "1.0 1.0" \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


# PAct adaptation

bash local/chain/adaptation/PAct/PAct_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-init 1.0 --param-alpha-init 0.0 \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk


# BPAct adaptation

bash local/chain/adaptation/PAct/BPAct_adaptation.sh \
 --baseline cnn_tdnn_1a_hires_sp \
 --adapt-ivector-dir exp/nnet3/ivectors_eval2000 \
 --test-ivector-dir exp/nnet3/ivectors_eval2000 \
 --adapted-layer "cnn1 tdnnf7 tdnnf8 tdnnf9 tdnnf10 tdnnf11 tdnnf12" \
 --layer-dim "2560 1536 1536 1536 1536 1536 1536" \
 --KL-scale "0.0001 1.0 1.0 1.0 1.0 1.0 1.0" \
 --input-config "component-node name=idct component=idct input=feature1" \
 --tag "_eval2000" \
 --epoch-num 7 --lr1 0.01 --lr2 0.01 --num-chunk 64 --param-alpha-mean-init 1.0 --param-alpha-std-init 1.0 --param-beta-mean-init 0.0 --param-beta-std-init 1.0 \
 --prior-alpha-mean "1.0 1.0" --prior-alpha-std "1.0 1.0" --prior-beta-mean "0.0 0.0" --prior-beta-std "1.0 1.0" \
 eval2000_hires_spk \
 exp/chain/cnn_tdnn_1a_hires_sp/decode_eval2000_hires_sw1_fsh_fg/1BEST_lat/score_10_0.0 \
 eval2000_hires_spk

