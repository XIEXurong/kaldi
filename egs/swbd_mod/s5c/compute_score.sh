dir=$1 # exp/chain/tdnn7r_sp/decode_eval2000_sw1_fsh_fg

. ./path.sh
. ./utils/parse_options.sh


grep Sum $dir/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh
grep Sum $dir/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh
grep Sum $dir/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh
grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh
