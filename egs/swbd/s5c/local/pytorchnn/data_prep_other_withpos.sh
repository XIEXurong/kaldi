#!/usr/bin/env bash


type="_phone"

. ./path.sh
. ./cmd.sh
. utils/parse_options.sh


lang=$1 # data/lang_nosp
indir=$2 # data/pytorchnn

dir=${indir}${type}
mkdir -p $dir/config

for text in train valid test; do
  cat $indir/${text}.txt | sed "s/^/NAME /g" | \
    steps/nnet3/chain/e2e/text_to_phones.py --edge-silprob 0 --between-silprob 0 $lang | sed "s/^NAME //g" > $dir/${text}.txt
done

echo "<unk>" >$dir/config/oov.txt
cp $lang/phones.txt $dir/words.txt
if ! grep -w '<unk>' $dir/words.txt >/dev/null; then
  n=$(cat $dir/words.txt | wc -l)
  echo "<unk> $n" >> $dir/words.txt
fi

echo "Data preparation done."

