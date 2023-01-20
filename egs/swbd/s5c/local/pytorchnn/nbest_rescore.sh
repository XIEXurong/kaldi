#!/usr/bin/env bash

# This script is very similar to steps/rnnlmrescore.sh, and it performs n-best
# LM rescoring with Kaldi-RNNLM.

# Begin configuration section.
N=10
inv_acwt=10
cmd=run.pl
stage=1 # Stage of this script, for partial reruns.
skip_scoring=false
keep_ali=true
scoring_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# != 4 ]; then
   echo "Do language model rescoring of lattices (partially remove old LM, add new LM)"
   echo "This version applies an RNNLM and mixes it with the LM scores"
   echo "previously in the lattices., controlled by the first parameter (rnnlm-weight)"
   echo ""
   echo "Usage: $0 [options] <rnn-weight> <old-lang-dir> <rnn-dir> <data-dir> <input-decode-dir> <output-decode-dir>"
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
data=$2
indir=$3
dir=$4

acwt=`perl -e "print (1.0/$inv_acwt);"`

for f in $indir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

nj=`cat $indir/num_jobs` || exit 1;
mkdir -p $dir;
cp $indir/num_jobs $dir/num_jobs

adir=$dir/archives

rm $dir/.error 2>/dev/null
mkdir -p $dir/log

# First convert lattice to N-best.  Be careful because this
# will be quite sensitive to the acoustic scale; this should be close
# to the one we'll finally get the best WERs with.
# Note: the lattice-rmali part here is just because we don't
# need the alignments for what we're doing.
if [ $stage -le 1 ]; then
  echo "$0: converting lattices to N-best lists."
  if $keep_ali; then
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.gz|" \
      "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.gz|" ark:- \|  \
      lattice-rmali ark:- "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  fi
fi

if [ $stage -le 3 ]; then
# Decompose the n-best lists into 4 archives.
  echo "$0: creating separate-archive form of N-best lists."
  $cmd JOB=1:$nj $dir/log/make_new_archives.JOB.log \
    mkdir -p $adir.JOB '&&' \
    nbest-to-linear "ark:gunzip -c $dir/nbest1.JOB.gz|" \
    "ark,t:$adir.JOB/ali" "ark,t:$adir.JOB/words" \
    "ark,t:$adir.JOB/lmwt.withlm" "ark,t:$adir.JOB/acwt" || exit 1;
fi

if [ $stage -le 8 ]; then
  echo "$0: reconstructing archives back into lattices."
  $cmd JOB=1:$nj $dir/log/reconstruct_lattice.JOB.log \
    linear-to-nbest "ark:$adir.JOB/ali" "ark:$adir.JOB/words" \
    "ark:$adir.JOB/lmwt.withlm" "ark:$adir.JOB/acwt" ark:- \| \
    nbest-to-lattice ark:- "ark:|gzip -c >$dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $oldlang $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;

