#!/usr/bin/env bash
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)
#           2021  Ke Li

# Begin configuration section.

LM=sw1_fsh_fg # Using the 4-gram const arpa file as old lm
decode_dir_suffix_forward=pytorch_transformer
decode_dir_suffix_backward=pytorch_transformer_back
pytorch_path=exp/pytorchnn_lm/pytorch_transformer_back
nn_model=
decode_dir=

model_type=Transformer # LSTM, GRU or Transformer
embedding_dim=512 # 650 for LSTM (for reproducing perplexities and WERs above)
hidden_dim=512 # 650 for LSTM
nlayers=6 # 2 for LSTM
nhead=8

weight=0.8
nbest_num=20
beam=5
beam_forward=
epsilon=0.5

seq_len=100
reset_history=true

cross_utt=false

data=data
LM_path=data/lang_

use_nbest=false
scoring_opts=
other_opt=
other_opt1=
tied=true

gpu_nj=

cuda_id="0,1,2,3,4,5,6,7,8"

use_gpu=false

data_dir=data/pytorchnn_back

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh
[ -z "$cmd" ] && cmd=$decode_cmd

set -e

if [ -z $nn_model ]; then
  nn_model=$pytorch_path/model.pt
fi

test_data=$1
ac_model_dir=$2

decode_dir_ori=$decode_dir

beam_forward1=${beam_forward:+${beam_forward}-}
if [ -z $beam_forward ]; then
    beam_forward=$beam
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

GPU=""
DEV="_cpu"
if [ "$use_gpu" == "true" ]; then GPU="_gpu"; DEV="_gpu"; fi

cross_utt_opt=""
CROSSUTT=""
if [ "$cross_utt" == "true" ]; then CROSSUTT="_crossutt"; cross_utt_opt="--seq_len $seq_len --reset_history $reset_history"; fi

if $use_nbest; then
  echo "$0: Perform N-best rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in $test_data; do
      if [ -z $decode_dir_ori ]; then
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      fi
      steps/pytorchnn/lmrescore_nbest_pytorchnn_back_mod${CROSSUTT}${GPU}.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt $other_opt1 \
        --cmd "$cmd --mem 4G" ${gpu_nj:+ --gpu_nj "$gpu_nj"} \
        --N $nbest_num \
        --model-type $model_type \
        --embedding_dim $embedding_dim \
        --hidden_dim $hidden_dim \
        --nlayers $nlayers \
        --nhead $nhead \
        --weight $weight \
        --tied $tied $cross_utt_opt \
        ${LM_path}$LM $nn_model $data_dir/words.txt \
        $data/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_${nbest_num}best_${weight} \
        ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}
        
      > ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${nbest_num}best_${weight}/scoring_all
  done
fi

if ! $use_nbest; then
  echo "$0: Perform lattice rescoring on $ac_model_dir with a $model_type LM."
  for decode_set in $test_data; do
      if [ -z $decode_dir_ori ]; then
        decode_dir=${ac_model_dir}/decode_${decode_set}_${LM}
      fi
      steps/pytorchnn/lmrescore_lattice_pytorchnn_back_mod.sh ${scoring_opts:+ --scoring-opts "$scoring_opts"} $other_opt \
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
        $data/${decode_set} ${decode_dir}_${decode_dir_suffix_forward}_${weight}_beam${beam_forward}_epsilon${epsilon} \
        ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}
        
      > ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/score_*/*.ctm.swbd.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/score_*/*.ctm.callhm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/score_*/*.ctm.fsh.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/scoring_all
      grep Sum ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/score_*/*.ctm.filt.sys | utils/best_wer.sh >> ${decode_dir}_${decode_dir_suffix_backward}_${weight}_beam${beam_forward1}${beam}_epsilon${epsilon}/scoring_all
  done
fi

exit 0
