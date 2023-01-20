
lm_scale=10
penalty=0.0
scale_opts="--transition-scale=0.0 --self-loop-scale=0.0"
lang=data/lang_e2e

best_dir=1BEST_fst


. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

dir=$1 # exp/chain/tdnn_fbk_iv_7q_subN_decode/decode_eval2000_fbk_spk${split}_sw1_fg_htk_arpa

num=`cat $dir/num_jobs`
for i in `seq 1 $num`; do
mkdir -p $dir/$best_dir/score_${lm_scale}_${penalty} && lattice-scale --lm-scale=$lm_scale \
 "ark:gunzip -c $dir/lat.$i.gz|" ark:- | \
 lattice-add-penalty --word-ins-penalty=$penalty ark:- ark:- | \
 lattice-best-path ark:- "ark,t:|int2sym.pl -f 2- $lang/words.txt > $dir/$best_dir/score_${lm_scale}_${penalty}/text.$i" || exit 1;
 
 oov_sym=`cat $lang/oov.int` || exit 1;
 
 compile-train-graphs $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    $dir/../tree $dir/../final.mdl $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $dir/$best_dir/score_${lm_scale}_${penalty}/text.$i |" \
    "ark,scp:$dir/$best_dir/score_${lm_scale}_${penalty}/fst.$i.ark,$dir/$best_dir/score_${lm_scale}_${penalty}/fst.$i.scp" || exit 1;
done
echo $num > $dir/$best_dir/score_${lm_scale}_${penalty}/num_jobs






