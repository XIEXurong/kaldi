#!/usr/bin/env bash

stage=0
lm_scale_list=
cmd=run.pl
skip_scoring=false
nnlm_score_file_name=lmwt.rnn
scoring_opt=
graph_scale=1

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

dir=$1
LM_list=$2
data=$3
oldlang=$4

LM_array=($LM_list)
LM_num=`echo ${#LM_array[*]}`
LM_index_max=$(awk "BEGIN{print($LM_num-1)}")
LM_num_include_base=$(awk "BEGIN{print($LM_num+1)}") # LM_base + all LMs

if [ -z "$lm_scale_list" ]; then
    LM_weight=$(awk "BEGIN{print(1/$LM_num_include_base)}") # average = 1/(LM_num+1)
    lm_scale_array=()
    for i in `seq 0 $LM_num`; do
        lm_scale_array[$i]=$LM_weight
    done
else
    lm_scale_array=($lm_scale_list)
    lm_scale_num=`echo ${#lm_scale_array[*]}`
    [[ "$lm_scale_num" == "$LM_num_include_base" ]] || exit 1;
fi

adir=$dir/archives
mkdir -p $adir

base_dir=${LM_array[0]}
nj=`cat $base_dir/num_jobs` || exit 1;

if [ $stage -le 1 ]; then
    echo "$0: Copying needed information to $adir"
    for n in `seq $nj`; do
        mkdir -p $adir.$n
        cp $base_dir/archives.$n/ali $adir.$n/
        cp $base_dir/archives.$n/words $adir.$n/
        cp $base_dir/archives.$n/words_text $adir.$n/
        cp $base_dir/archives.$n/lmwt.nolm $adir.$n/
        cp $base_dir/archives.$n/acwt $adir.$n/
        cp $base_dir/archives.$n/lmwt.withlm $adir.$n/
        paste $adir.$n/lmwt.nolm $adir.$n/lmwt.withlm | awk '{print $1, ($4-$2);}' > \
          $adir.$n/lmwt.lmonly || exit 1;

        for i in `seq 0 $LM_index_max`; do
            cp ${LM_array[$i]}/archives.$n/$nnlm_score_file_name $adir.$n/lmwt.rnn_$i
        done
    done
fi

if [ $stage -le 2 ]; then
    echo "$0: reconstructing total LM+graph scores including interpolation of RNNLM and old LM scores."
    
    for n in `seq $nj`; do
        mkdir -p $adir.$n/temp
        lmweight=${lm_scale_array[0]}
        paste $adir.$n/lmwt.nolm $adir.$n/lmwt.lmonly | awk -v lmweight=$lmweight -v gweight=$graph_scale \
          '{ key=$1; graphscore=$2; lmscore=$4;
         score = (gweight*graphscore)+(lmweight*lmscore);
         print $1,score; } ' > $adir.$n/temp/lmwt.base0 || exit 1;
        for i in `seq 0 $LM_index_max`; do
            j=$(awk "BEGIN{print($i+1)}")
            lmweight=${lm_scale_array[j]}
            paste $adir.$n/temp/lmwt.base$i $adir.$n/lmwt.rnn_$i | awk -v lmweight=$lmweight \
              '{ key=$1; graphscore=$2; lmscore=$4;
             score = graphscore+(lmweight*lmscore);
             print $1,score; } ' > $adir.$n/temp/lmwt.base$j || exit 1;
        done
        cp $adir.$n/temp/lmwt.base$j $adir.$n/lmwt.interp
    done
fi

if [ $stage -le 3 ]; then
  echo "$0: reconstructing archives back into lattices."
  $cmd JOB=1:$nj $dir/log/reconstruct_lattice.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" \
    "ark:$adir.JOB/lmwt.interp" "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opt $data $oldlang $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
    > $dir/scoring_all
    grep Sum $dir/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> $dir/scoring_all
    grep Sum $dir/score_*/*.ctm.filt.sys | utils/best_wer.sh >> $dir/scoring_all
fi

exit 0;