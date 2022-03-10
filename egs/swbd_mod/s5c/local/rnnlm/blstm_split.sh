#!/usr/bin/env bash

# Begin configuration section.

embedding_dim=1024

. ./path.sh
. ./utils/parse_options.sh

dir=$1
dir_forward=${dir}/forward
dir_backward=${dir}/backward

# forward
mkdir -p $dir_forward/config
cp $dir/config/words.txt $dir_forward/config/words.txt
cp $dir/special_symbol_opts.txt $dir_forward/special_symbol_opts.txt

cat <<EOF > $dir_forward/config/change.config
	component name=zeros type=ConstantFunctionComponent input-dim=$embedding_dim is-updatable=false output-dim=$embedding_dim output-mean=0 output-stddev=0
	component-node name=zeros component=zeros input=input
    component-node name=output.affine component=output.affine input=Append(tdnn3.renorm, zeros)
    
    # 2 times of output
    output-node name=output input=Scale(2.0, output.affine) objective=linear
EOF

nnet3-copy --nnet-config=$dir_forward/config/change.config $dir/final.raw - | nnet3-copy --edits="remove-orphans" - $dir_forward/final.raw
nnet3-info $dir_forward/final.raw > $dir_forward/final.info

ln -s $PWD/$dir/feat_embedding.final.mat $dir_forward/feat_embedding.final.mat
ln -s $PWD/$dir/word_feats.txt $dir_forward/word_feats.txt


# backforward

mkdir -p $dir_backward/config
cp $dir/config/words.txt $dir_backward/config/words.txt
cp $dir/special_symbol_opts.txt $dir_backward/special_symbol_opts.txt

cat <<EOF > $dir_backward/config/bias.config
    component name=zeros type=ConstantFunctionComponent input-dim=$embedding_dim is-updatable=false output-dim=$embedding_dim output-mean=0 output-stddev=0
	component-node name=zeros component=zeros input=input
    component-node name=output.affine component=output.affine input=Append(input, zeros)
EOF

nnet3-copy --nnet-config=$dir_backward/config/bias.config $dir/final.raw - | nnet3-copy --binary=false --edits="remove-orphans" - - | \
 grep "<BiasParams>" | sed "s/<BiasParams>//g" > $dir_backward/output.bias

cat <<EOF > $dir_backward/config/change.config
    component-node name=tdnn1_backward.affine component=tdnn1_backward.affine input=Append(input, IfDefined(Offset(input, -1)))
    component-node name=lstm1_backward.W_all component=lstm1_backward.W_all input=Append(tdnn1_backward.renorm, IfDefined(Offset(lstm1_backward.r_trunc, -1)))
    component-node name=lstm1_backward.lstm_nonlin component=lstm1_backward.lstm_nonlin input=Append(lstm1_backward.W_all, IfDefined(Offset(lstm1_backward.c_trunc, -1)))
    component-node name=tdnn2_backward.affine component=tdnn2_backward.affine input=Append(lstm1_backward.rp, IfDefined(Offset(lstm1_backward.rp, -3)))
    component-node name=lstm2_backward.W_all component=lstm2_backward.W_all input=Append(tdnn2_backward.renorm, IfDefined(Offset(lstm2_backward.r_trunc, -1)))
    component-node name=lstm2_backward.lstm_nonlin component=lstm2_backward.lstm_nonlin input=Append(lstm2_backward.W_all, IfDefined(Offset(lstm2_backward.c_trunc, -1)))
    component-node name=tdnn3_backward.affine component=tdnn3_backward.affine input=Append(lstm2_backward.rp, IfDefined(Offset(lstm2_backward.rp, -3)))
    
    component name=zeros type=ConstantFunctionComponent input-dim=$embedding_dim is-updatable=false output-dim=$embedding_dim output-mean=0 output-stddev=0
	component-node name=zeros component=zeros input=input
    component-node name=output.affine component=output.affine input=Append(zeros, tdnn3_backward.renorm)
    
    component name=output.bias type=ConstantFunctionComponent input-dim=$embedding_dim is-updatable=false output-dim=$embedding_dim vector=$dir_backward/output.bias
	component-node name=output.bias component=output.bias input=input
    
    # 2 times of output, no bias
    output-node name=output input=Scale(2.0, Sum(output.affine, Scale(-1, output.bias))) objective=linear
EOF

nnet3-copy --nnet-config=$dir_backward/config/change.config $dir/final.raw - | nnet3-copy --edits="remove-orphans" - $dir_backward/final.raw
nnet3-info $dir_backward/final.raw > $dir_backward/final.info

ln -s $PWD/$dir/feat_embedding.final.mat $dir_backward/feat_embedding.final.mat
ln -s $PWD/$dir/word_feats.txt $dir_backward/word_feats.txt


