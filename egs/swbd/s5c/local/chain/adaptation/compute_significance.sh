
tool_base=/home/xxr/works/kaldi_bayes_adapt/tools/sctk/bin

baseline_score_file=$1 # exp/chain/e2e_tdnnf_7r_bi_mmice/scoring_all
score_file=$2 # exp/chain/adaptation/LHUC_e2e/e2e_tdnnf_7r_bi_mmice_BLHUC_e2e_eval2000_e2ehires_mmice_adaptlayer14_actSig_epoch7_lr10.1_lr20.1/scoring_all
lang=$3 # data/lang_sw1_fsh_fg

. ./path.sh

[ -f `dirname $score_file`/scoring_all_sig ] && echo "Have found the scoring_all_sig file." && exit 0

new_score_file=`tail -n 1 $score_file | sed "s/^.*exp/exp/g"`
data_name=`basename $new_score_file | sed "s/\..*$//g"`
new_score=`echo $new_score_file | sed "s/\.[^\.]*$//g"`
score_dir=$(dirname $new_score)

if [ ! -f $new_score_file ]; then
    dir=$(dirname $score_dir)
    lms=`basename $score_dir | sed "s/_/ /g" | awk '{print $2}'`
    ip=`basename $score_dir | sed "s/_/ /g" | awk '{print $3}'`
    bash local/score.sh --min_lmwt $lms --max_lmwt $lms --word_ins_penalty $ip data/$data_name $lang $dir
fi

if [ ! -f ${new_score}.sgml ]; then
    $tool_base/sclite -r $score_dir/stm.filt stm -h $score_dir/$data_name.ctm.filt ctm $score_dir/$data_name.ctm -F -D -o sgml -C det sbhist hist -O $score_dir -n $data_name.ctm.filt &> $score_dir/../scoring/log/sgml.log
fi

base_score_file=`tail -n 1 $baseline_score_file | sed "s/^.*exp/exp/g"`
data_name=`basename $base_score_file | sed "s/\..*$//g"`
base_score=`echo $base_score_file | sed "s/\.[^\.]*$//g"`
baseline_score_dir=$(dirname $base_score)

if [ ! -f $base_score_file ]; then
    baseline_dir=$(dirname $baseline_score_dir)
    lms=`basename $baseline_score_dir | sed "s/_/ /g" | awk '{print $2}'`
    ip=`basename $baseline_score_dir | sed "s/_/ /g" | awk '{print $3}'`
    bash local/score.sh --min_lmwt $lms --max_lmwt $lms --word_ins_penalty $ip data/$data_name $lang $baseline_dir
fi

if [ ! -f ${base_score}.sgml ]; then
    $tool_base/sclite -r $baseline_score_dir/stm.filt stm -h $baseline_score_dir/$data_name.ctm.filt ctm $baseline_score_dir/$data_name.ctm -F -D -o sgml -C det sbhist hist -O $baseline_score_dir -n $data_name.ctm.filt &> $score_dir/../scoring/log/sgml.log
fi

cat ${base_score}.sgml ${new_score}.sgml | /home/xxr/works/kaldi_bayes_adapt/tools/sctk/bin/sc_stats -p -t mapsswe -v -u -n score_sctk_tmp.test
grep "|   MP    ||" score_sctk_tmp.test.stats.unified > `dirname $score_file`/scoring_all_sig
