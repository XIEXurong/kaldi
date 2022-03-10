#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# Begin configuration section.

LM=sw1_fsh_fg # Using the 4-gram const arpa file as old lm
decode_dir_suffix=pytorch_transformer
pytorch_path=exp/pytorchnn_lm/pytorch_transformer
nn_model=

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 650 for LSTM (for reproducing perplexities and WERs above)
hidden_dim=512 # 650 for LSTM
nlayers=6 # 2 for LSTM
nhead=8

weight=0.8
nbest_num=20
beam=5
epsilon=0.5

data=data
LM_path=data/lang_

use_nbest=false
scoring_opts=
other_opt=
tied=true

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

if [ -z $nn_model ]; then
  nn_model=$pytorch_path/model.pt
fi

data_dir=data/pytorchnn
test_data=$1
ac_model_dir=$2

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

if $use_nbest; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in $test_data; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      steps/pytorchnn/lmrescore_nbest_pytorchnn.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
        --cmd "$cmd --mem 4G" \
        --N $nbest_num \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight $weight \
        --tied $tied \
        ${LM_path}$LM $nn_model $data_dir/words.txt \
        $data/${decode_set} ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}
        
      > ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${nbest_num}best_${weight}/scoring_all
  done
fi

if ! $use_nbest; then
  echo "$0: Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in $test_data; do
      decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      steps/pytorchnn/lmrescore_lattice_pytorchnn.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
        --cmd "$cmd --mem 4G" \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight $weight \
        --beam $beam \
        --epsilon $epsilon \
        --tied $tied \
        ${LM_path}$LM $nn_model $data_dir/words.txt \
        $data/${decode_set} ${decode_dir} \
        ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}
        
      > ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix}_${weight}_beam${beam}_epsilon${epsilon}/scoring_all
  done
fi

exit 0
