
scale_opts="--transition-scale=0.0 --self-loop-scale=0.0"
lang=data/lang_e2e
tag=

num=50

best_dir=1BEST_fst


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/cnn_tdnn1a_specaugkaldi_sp
data=$2 # data/eval2000_hires

mkdir -p $dir/fst_oracle${tag}/$best_dir
for i in `seq 1 $num`; do
 cat $data/split${num}/$i/text | awk '{printf $1; for(n=2;n<=NF;n++) printf(" %s", tolower($n)); printf("\n")}' > $dir/fst_oracle${tag}/$best_dir/text.$i
 
 oov_sym=`cat $lang/oov.int` || exit 1;
 
 compile-train-graphs $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    $dir/tree $dir/final.mdl $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $dir/fst_oracle${tag}/$best_dir/text.$i |" \
    "ark,scp:$dir/fst_oracle${tag}/$best_dir/fst.$i.ark,$dir/fst_oracle${tag}/$best_dir/fst.$i.scp" || exit 1;
done
echo $num > $dir/fst_oracle${tag}/$best_dir/num_jobs






