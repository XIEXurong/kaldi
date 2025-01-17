#!/bin/bash

# Copyright 2017   Hainan Xu
# Apache 2.0

# This script rescores lattices with KALDI RNNLM using a pruned algorithm.
# The details of the algorithm could be found at
# http://www.danielpovey.com/files/2018_icassp_lattice_pruning.pdf
# One example script for this is at egs/swbd/s5c/local/rnnlm/run_lstm.sh

# Begin configuration section.
cmd=run.pl
srun_cmd="srun -p pBatch_level2"
num_jobs=
skip_scoring=false
max_ngram_order=4 # Approximate the lattice-rescoring by limiting the max-ngram-order
                  # if it's set, it merges histories in the lattice if they share
                  # the same ngram history and this prevents the lattice from 
                  # exploding exponentially. Details of the n-gram approximation
                  # method are described in section 2.3 of the paper
                  # http://www.danielpovey.com/files/2018_icassp_lattice_pruning.pdf
max_arcs=         # limit the max arcs in lattice while rescoring. E.g., 20000

acwt=0.1
weight=0.5  # Interpolation weight for RNNLM.
normalize=false # If true, we add a normalization step to the output of the RNNLM
                # so that it adds up to *exactly* 1. Note that this is not necessary
                # as in our RNNLM setup, a properly trained network would automatically
                # have its normalization term close to 1. The details of this
                # could be found at http://www.danielpovey.com/files/2018_icassp_rnnlm.pdf
lattice_prune_beam=8 # Beam used in pruned lattice composition
                     # This option affects speed and how large the composed lattice may be
scoring_opts=

# End configuration section.

echo "$0 $@"  # Print the command line for logging

. ./utils/parse_options.sh

if [ $# != 5 ]; then
   echo "Does language model rescoring of lattices (remove old LM, add new LM)"
   echo "with Kaldi RNNLM using a pruned algorithm. See comments in file for details"
   echo ""
   echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> \\"
   echo "                   <data-dir> <input-decode-dir> <output-decode-dir>"
   echo " e.g.: $0 data/lang_tg exp/rnnlm_lstm/ data/test \\"
   echo "                   exp/tri3/test_tg exp/tri3/test_rnnlm_4gram"
   echo "options: [--cmd (run.pl|queue.pl [queue opts])]"
   exit 1;
fi

[ -f path.sh ] && . ./path.sh;

oldlang=$1
rnnlm_dir=$2
data=$3
indir=$4
outdir=$5

oldlm=$oldlang/G.fst
carpa_option=
if [ ! -f $oldlm ]; then
  echo "$0: file $oldlm not found; looking for $oldlang/G.carpa"
  oldlm=$oldlang/G.carpa
  carpa_option="--use-const-arpa=true"
fi

[ ! -f $oldlm ] && echo "$0: Missing file $oldlm" && exit 1;
[ ! -f $rnnlm_dir/final.raw ] && echo "$0: Missing file $rnnlm_dir/final.raw" && exit 1;
[ ! -f $rnnlm_dir/feat_embedding.final.mat ] && [ ! -f $rnnlm_dir/word_embedding.final.mat ] && echo "$0: Missing word embedding file" && exit 1;

[ ! -f $oldlang/words.txt ] &&\
  echo "$0: Missing file $oldlang/words.txt" && exit 1;
! ls $indir/lat.*.gz >/dev/null &&\
  echo "$0: No lattices input directory $indir" && exit 1;
awk -v n=$0 -v w=$weight 'BEGIN {if (w < 0 || w > 1) {
  print n": Interpolation weight should be in the range of [0, 1]"; exit 1;}}' \
  || exit 1;

if ! head -n -1 $rnnlm_dir/config/words.txt | cmp $oldlang/words.txt -; then
  # the last word of the RNNLM word list is an added <brk> word
  echo "$0: Word lists mismatch for lattices and RNNLM."
  exit 1
fi

normalize_opt=
if $normalize; then
  normalize_opt="--normalize-probs=true"
fi
special_symbol_opts=$(cat $rnnlm_dir/special_symbol_opts.txt)

word_embedding=
if [ -f $rnnlm_dir/word_embedding.final.mat ]; then
  word_embedding=$rnnlm_dir/word_embedding.final.mat
else
  word_embedding="rnnlm-get-word-embedding $rnnlm_dir/word_feats.txt $rnnlm_dir/feat_embedding.final.mat -|"
fi

max_arcs_opt=
if [ ! -z "$max_arcs" ]; then
  max_arcs_opt="--max-arcs=$max_arcs"
fi

mkdir -p $outdir/log
if [ ! -z "$num_jobs" ]; then
	nj=$num_jobs
	echo "$num_jobs" > $outdir/num_jobs || exit 1;
	
	mkdir -p $outdir/old_lat
	nj_old=`cat $indir/num_jobs` || exit 1;
	for JOB in `seq 1 $nj_old`; do
		lattice-push --push-strings=false --push-weights=false "ark:gunzip -c $indir/lat.$JOB.gz|" ark,scp:$outdir/old_lat/$JOB.lat,$outdir/old_lat/$JOB.lat.scp &
	done
	wait
	
	cat $outdir/old_lat/*.lat.scp > $outdir/old_lat/all_lat.scp
	splitFile.sh $outdir/old_lat/all_lat.scp $nj
	
	for JOB in `seq 1 $nj`; do
	  ( $srun_cmd lattice-lmrescore-kaldi-rnnlm-pruned --lm-scale=$weight $special_symbol_opts \
		--lattice-compose-beam=$lattice_prune_beam \
		--acoustic-scale=$acwt --max-ngram-order=$max_ngram_order $normalize_opt $max_arcs_opt \
		$carpa_option $oldlm "$word_embedding" "$rnnlm_dir/final.raw" \
		scp:$outdir/old_lat/all_lat_sub$JOB.scp "ark,t:|gzip -c>$outdir/lat.$JOB.gz" &> $outdir/log/rescorelm.$JOB.log || exit 1; ) &
	done
	wait
else
	nj=`cat $indir/num_jobs` || exit 1;
	cp $indir/num_jobs $outdir
	
	for JOB in `seq 1 $nj`; do
	  ( $srun_cmd lattice-lmrescore-kaldi-rnnlm-pruned --lm-scale=$weight $special_symbol_opts \
		--lattice-compose-beam=$lattice_prune_beam \
		--acoustic-scale=$acwt --max-ngram-order=$max_ngram_order $normalize_opt $max_arcs_opt \
		$carpa_option $oldlm "$word_embedding" "$rnnlm_dir/final.raw" \
		"ark:gunzip -c $indir/lat.$JOB.gz|" "ark,t:|gzip -c>$outdir/lat.$JOB.gz" &> $outdir/log/rescorelm.$JOB.log || exit 1; ) &
	done
	wait
fi

if ! $skip_scoring ; then
  err_msg="$0: Not scoring because local/score.sh does not exist or not executable."
  [ ! -x local/score.sh ] && echo $err_msg && exit 1;
  echo local/score.sh --cmd "$cmd" $scoring_opts $data $oldlang $outdir
  local/score.sh --cmd "$cmd" $scoring_opts $data $oldlang $outdir
else
  echo "$0: Not scoring because --skip-scoring was specified."
fi

exit 0;
