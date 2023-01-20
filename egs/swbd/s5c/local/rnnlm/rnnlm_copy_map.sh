#!/usr/bin/env bash

new_word_list=

. ./utils/parse_options.sh

old_dir=$1
new_dir=$2

mkdir -p $new_dir

for f in `ls $old_dir`; do
    ln -s $PWD/$old_dir/$f $new_dir/$f
done


if [ ! -z $new_word_list ]; then
    rm $new_dir/config
    cp -r $old_dir/config $new_dir/config

    rm $new_dir/word_feats.txt
    cp -r $old_dir/word_feats.txt $new_dir/word_feats.txt
    
    python3 local/rnnlm/change_word_id.py $new_dir $new_word_list
    mv $new_dir/config/words.txt $new_dir/config/words_old.txt
    cp $new_word_list $new_dir/config/words.txt
    mv $new_dir/word_feats.txt $new_dir/word_feats_old.txt
    mv $new_dir/word_feats_new.txt $new_dir/word_feats.txt
fi

