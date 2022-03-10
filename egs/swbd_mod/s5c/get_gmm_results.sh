for x in exp/tri4/decode_eval2000*; do
 [ -d $x ] && grep Sum $x/score_*/eval2000.ctm.swbd.filt.sys | utils/best_wer.sh
 [ -d $x ] && grep Sum $x/score_*/eval2000.ctm.callhm.filt.sys | utils/best_wer.sh
 [ -d $x ] && grep Sum $x/score_*/eval2000.ctm.filt.sys | utils/best_wer.sh
done > exp/tri4/scoring_all
