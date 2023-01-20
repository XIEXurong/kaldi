#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# Begin configuration section.
stage=0

pytorch_path=exp/pytorchnn_lm/pytorch_transformer
nn_model=

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 650 for LSTM (for reproducing perplexities and WERs above)
hidden_dim=512 # 650 for LSTM
nlayers=6 # 2 for LSTM
nhead=8
learning_rate=0.1 # 5 for LSTM
seq_len=100
dropout=0.1
tied=true

data_dir=data/pytorchnn

limit_num_gpus_cmd="utils/parallel/limit_num_gpus.sh"

cuda_id="0,1,2,3,4,5,6,7,8"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

export CUDA_VISIBLE_DEVICES=$cuda_id

set -e

if [ -z $nn_model ]; then
  nn_model=$pytorch_path/model.pt
fi

# Check if PyTorch is installed to use with python
if python3 steps/pytorchnn/check_py.py 2>/dev/null; then
  echo PyTorch is ready to use on the python side. This is good.
else
  echo PyTorch not found on the python side.
  echo Please install PyTorch first. For example, you can install it with conda:
  echo "conda install pytorch torchvision cudatoolkit=10.2 -c pytorch", or
  echo with pip: "pip install torch torchvision". If you already have PyTorch
  echo installed somewhere else, you need to add it to your PATH.
  echo Note: you need to install higher version than PyTorch 1.1 to train Transformer models
  exit 1
fi

if [ $stage -le 0 ]; then
  local/pytorchnn/data_prep.sh $data_dir
fi

tied_opt=
if $tied; then
  tied_opt="--tied"
fi

if [ $stage -le 1 ]; then
  # Train a PyTorch neural network language model.
  echo "Start neural network language model training."
  $cuda_cmd $pytorch_path/log/train.log $limit_num_gpus_cmd \
    python3 steps/pytorchnn/train.py --data $data_dir \
            --model $model_type \
            --emsize $embedding_dim \
            --nhid $hidden_dim \
            --nlayers $nlayers \
            --nhead $nhead \
            --lr $learning_rate \
            --dropout $dropout \
            --seq_len $seq_len \
            --clip 1.0 \
            --batch-size 32 \
            --epoch 64 \
            --save $nn_model \
            --cuda \
            $tied_opt
fi

exit 0
