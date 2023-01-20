#!/usr/bin/env bash


type="_phone"

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh


lang=$1 # data/lang_nosp
indir=$2 # data/pytorchnn

dir=${indir}${type}
mkdir -p $dir/config/phones

echo "<unk>" > $dir/config/oov.txt
cat $lang/phones/align_lexicon.txt | sed "s/_B//g" | sed "s/_E//g" | sed "s/_I//g" | sed "s/_S//g" > $dir/config/phones/align_lexicon.txt
cat $lang/phones/optional_silence.txt > $dir/config/phones/optional_silence.txt

for text in train valid test; do
  cat $indir/${text}.txt | sed "s/^/NAME /g" | \
    steps/nnet3/chain/e2e/text_to_phones.py --edge-silprob 0 --between-silprob 0 $dir/config | sed "s/^NAME //g" > $dir/${text}.txt
done

cat $lang/phones.txt | awk '{print $1}' | sed "s/_B//g" | sed "s/_E//g" | sed "s/_I//g" | sed "s/_S//g" | uniq | python -c 'import sys
i=int(0)
for l in sys.stdin:
  w = l.strip().split(" ")[0]
  print(w + " " + str(i))
  i=i+1
' > $dir/words.txt

if ! grep -w '<unk>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<unk> $n" >> $dir/words.txt
fi

echo "Data preparation done."

