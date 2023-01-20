#!/usr/bin/env bash

# This script is very similar to rnnlm/lmrescore_nbest.sh, and it performs N-best
# LM rescoring with a Pytorch-trained neural LM.

# Begin configuration section.
scripts=local/chain/adaptation

N=10
model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=768
hidden_dim=768
nlayers=6
nhead=8

conv_sort_list=

seq_len=100

inv_acwt=10
weight=0.8 # interpolation weight of a neural network LM with a N-gram LM
oov_symbol="'<unk>'"

gpu_nj=1

cmd=run.pl
use_phi=false  # This is kind of an obscure option.  If true, we'll remove the old
  # LM weights (times 1-RNN_scale) using a phi (failure) matcher, which is
  # appropriate if the old LM weights were added in this way, e.g. by
  # lmrescore.sh.  Otherwise we'll use normal composition, which is appropriate
  # if the lattices came directly from decoding.  This won't actually make much
  # difference (if any) to WER, it's more so we know we are doing the right thing.
test=false # Activate a testing option.
stage=1 # Stage of this script, for partial reruns.
skip_scoring=false
keep_ali=true
scoring_opts=
tied=true
reset_history=true
gpu_wait=

limit_num_gpus_cmd="utils/parallel/limit_num_gpus.sh"

# End configuration section.

echo "$0 $*"  # Print the command line for logging

. ./cmd.sh
[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# != 6 ]; then
   echo "Perform N-best rescoring with a PyTorch-trained neural language model."
   echo "The neural LM is interpolated with an N-gram LM during rescoring."
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <nn-model-dir> vocab <data-dir> <input-decode-dir> <output-decode-dir>"
   echo "Main options:"
   echo "  --inv-acwt <inv-acwt>          # default 12.  e.g. --inv-acwt 17.  Equivalent to LM scale to use."
   echo "                                 # for N-best list generation... note, we'll score at different acwt's"
   echo "  --cmd <run.pl|queue.pl [opts]> # how to run jobs."
   echo "  --phi (true|false)             # Should be set to true if the source lattices were created"
   echo "                                 # by lmrescore.sh, false if they came from decoding."
   echo "  --N <N>                        # Value of N in N-best rescoring (default: 10)"
   exit 1;
fi

oldlang=$1
nn_model=$2
vocab=$3 # Vocabulary used for training the neural language model. This is
         # usually the same as $oldlang/words.txt.
data=$4
indir=$5
dir=$6

acwt=$(perl -e "print (1.0/$inv_acwt);")

# Figures out if the old LM is G.fst or G.carpa
oldlm=$oldlang/G.fst
if [ -f $oldlang/G.carpa ]; then
  oldlm=$oldlang/G.carpa
elif [ ! -f $oldlm ]; then
  echo "$0: expecting either $oldlang/G.fst or $oldlang/G.carpa to exist" &&\
    exit 1;
fi

for f in $nn_model $vocab $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

nj=$(cat $indir/num_jobs) || exit 1;
mkdir -p $dir;
cp $indir/num_jobs $dir/num_jobs

adir=$dir/archives

phi=$(grep -w '#0' $oldlang/words.txt | awk '{print $2}')

rm $dir/.error 2>/dev/null
mkdir -p $dir/log

# First convert lattice to N-best.  Be careful because this
# will be quite sensitive to the acoustic scale; this should be close
# to the one we'll finally get the best WERs with.
# Note: the lattice-rmali part here is just because we don't
# need the alignments for what we're doing.
if [ $stage -le 5 ]; then
  echo "$0: Copying needed information from $indir/archives to $adir"
    # Do some small tasks; for these we don't use the queue, it will only slow us down.
  for n in `seq $nj`; do
    mkdir -p $adir.$n
    cp $indir/archives.$n/ali $adir.$n/
    cp $indir/archives.$n/words $adir.$n/
    cp $indir/archives.$n/words_text $adir.$n/
    cp $indir/archives.$n/lmwt.nolm $adir.$n/
    cp $indir/archives.$n/acwt $adir.$n/
    cp $indir/archives.$n/lmwt.withlm $adir.$n/
    
    cat $adir.$n/words | awk '{printf("%s ",$1);for(i=NF;i>1;i--) printf("%s ",$i); print""}' > $adir.$n/words_inv
    utils/int2sym.pl -f 2- $oldlang/words.txt < $adir.$n/words_inv > $adir.$n/words_text_inv || exit 1;
    mkdir -p $adir.$n/temp
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.withlm | awk '{print $1, ($4-$2);}' > \
      $adir.$n/lmwt.lmonly || exit 1;
  done
fi

tied_opt=
if $tied; then
  tied_opt="--tied"
fi

reset_history_opt=
if $reset_history; then
  reset_history_opt="--reset_history"
fi

if [ $stage -le 6 ]; then
  echo "$0: computing neural LM scores of the N-best list in parallel for each lattice."
  mkdir -p ${adir}_tmp
  
  if [ -z $conv_sort_list ]; then
  
      if [ "$gpu_nj" == "1" ]; then
          cat $adir.*/words_text_inv > ${adir}_tmp/words_text_inv
          
          $cuda_cmd $dir/log/compute_sentence_scores.log $limit_num_gpus_cmd \
            python3 steps/pytorchnn/compute_sentence_scores_crossutt_gpu.py \
                --infile ${adir}_tmp/words_text_inv \
                --outfile ${adir}_tmp/lmwt.nn \
                --vocabulary $vocab \
                --model-path $nn_model \
                --model $model_type \
                --emsize $embedding_dim \
                --nhid $hidden_dim \
                --nlayers $nlayers \
                --nhead $nhead \
                --seq_len $seq_len \
                --oov "$oov_symbol" --cuda $tied_opt $reset_history_opt ${gpu_wait:+ --gpu_wait} || exit 1;
      elif [ "$gpu_nj" -gt "1" ]; then
          job_len=$(awk -v nj=$nj -v gpu_nj=$gpu_nj 'BEGIN{printf("%.0f",nj/gpu_nj)}')
          for i in `seq 1 $gpu_nj`; do
            id_start=$(awk "BEGIN{print(($i-1)*$job_len+1)}")
            if [ "$i" == "$gpu_nj" ]; then
              id_end=$nj
            else
              id_end=$(awk "BEGIN{print($i*$job_len)}")
            fi
            echo "Job $i: process the data from $id_start to $id_end."
            for j in `seq $id_start $id_end`; do
              cat $adir.$j/words_text_inv >> ${adir}_tmp/words_text_inv.$i
            done
            $cuda_cmd $dir/log/compute_sentence_scores.$i.log $limit_num_gpus_cmd \
                python3 steps/pytorchnn/compute_sentence_scores_crossutt_gpu.py \
                    --infile ${adir}_tmp/words_text_inv.$i \
                    --outfile ${adir}_tmp/lmwt.nn.$i \
                    --vocabulary $vocab \
                    --model-path $nn_model \
                    --model $model_type \
                    --emsize $embedding_dim \
                    --nhid $hidden_dim \
                    --nlayers $nlayers \
                    --nhead $nhead \
                    --seq_len $seq_len \
                    --oov "$oov_symbol" --cuda $tied_opt $reset_history_opt ${gpu_wait:+ --gpu_wait} || exit 1;
          done
          cat ${adir}_tmp/lmwt.nn.* > ${adir}_tmp/lmwt.nn
      fi

  else
      > ${adir}_tmp/words_text_inv
      for utt in `cat $conv_sort_list | awk '{print $1}'`; do
        cat $adir.*/words_text_inv | grep $utt >> ${adir}_tmp/words_text_inv
      done
      $cuda_cmd $dir/log/compute_sentence_scores.log $limit_num_gpus_cmd \
            python3 steps/pytorchnn/compute_sentence_scores_crossutt_gpu.py \
                --infile ${adir}_tmp/words_text_inv \
                --outfile ${adir}_tmp/lmwt.nn \
                --vocabulary $vocab \
                --model-path $nn_model \
                --model $model_type \
                --emsize $embedding_dim \
                --nhid $hidden_dim \
                --nlayers $nlayers \
                --nhead $nhead \
                --seq_len $seq_len \
                --oov "$oov_symbol" --cuda $tied_opt $reset_history_opt ${gpu_wait:+ --gpu_wait} --conversation_history || exit 1;
  fi

  for i in `seq 1 $nj`; do
    perl $scripts/find_pdf.pl ${adir}_tmp/lmwt.nn $adir.$i/words_text_inv > $adir.$i/lmwt.nn
  done
  rm -r ${adir}_tmp
fi

if [ $stage -le 7 ]; then
  echo "$0: reconstructing total LM+graph scores including interpolation of neural LM and old LM scores."
  for n in $(seq $nj); do
    < $adir.$n/lmwt.nn awk '{sum=0;for(i=2;i<=NF;i++)sum+=$i; print $1,sum}' > $adir.$n/lmwt.nn.sum
    
    paste $indir/archives.$n/lmwt.nn.sum $adir.$n/lmwt.nn.sum | awk -F' ' '{print $1,$2 * 0.5 + $4 * 0.5}' > $adir.$n/lmwt.nn.sum_bi
    
    paste $adir.$n/lmwt.nolm $adir.$n/lmwt.lmonly $adir.$n/lmwt.nn.sum_bi | awk -v nnweight=$weight \
      '{ key=$1; graphscore=$2; lmscore=$4; nnscore=$6;
     score = graphscore+(nnweight*nnscore)+((1-nnweight)*lmscore);
     print $1,score; } ' > $adir.$n/lmwt.interp || exit 1;
  done
fi

if [ $stage -le 8 ]; then
  echo "$0: reconstructing archives back into lattices."
  $cmd JOB=1:$nj $dir/log/reconstruct_lattice.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" \
    "ark:$adir.JOB/lmwt.interp" "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  echo "scoring..."
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $oldlang $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;

