from __future__ import division
from __future__ import print_function

import logging
import math
import os
import sys

import libs.common as common_lib
import libs.nnet3.train.common as common_train_lib
import libs.nnet3.train.chain_objf.acoustic_model as chain_lib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def generate_chain_egs_adapt(dir, data, lat_dir, egs_dir,
                       left_context, right_context,
                       run_opts, stage=0,
                       left_tolerance=None, right_tolerance=None,
                       left_context_initial=-1, right_context_final=-1,
                       frame_subsampling_factor=3,
                       alignment_subsampling_factor=3,
                       online_ivector_dir=None,
                       frames_per_iter=20000, frames_per_eg_str="20", srand=0, deriv_weights_scp=None,
                       egs_opts=None, cmvn_opts=None):
    """Wrapper for steps/nnet3/chain/get_egs.sh

    See options in that script.
    """

    common_lib.execute_command(
        """steps/nnet3/chain/get_egs_noexclude.sh {egs_opts} \
                --cmd "{command}" \
                --cmvn-opts "{cmvn_opts}" \
                --online-ivector-dir "{ivector_dir}" \
                --left-context {left_context} \
                --right-context {right_context} \
                --left-context-initial {left_context_initial} \
                --right-context-final {right_context_final} \
                --left-tolerance '{left_tolerance}' \
                --right-tolerance '{right_tolerance}' \
                --frame-subsampling-factor {frame_subsampling_factor} \
                --alignment-subsampling-factor {alignment_subsampling_factor} \
                --stage {stage} \
                --frames-per-iter {frames_per_iter} \
                --frames-per-eg {frames_per_eg_str} \
                --srand {srand} \
                --deriv-weights-scp "{deriv_weights_scp}" \
                {data} {dir} {lat_dir} {egs_dir}""".format(
                    command=run_opts.egs_command,
                    cmvn_opts=cmvn_opts if cmvn_opts is not None else '',
                    ivector_dir=(online_ivector_dir
                                 if online_ivector_dir is not None
                                 else ''),
                    left_context=left_context,
                    right_context=right_context,
                    left_context_initial=left_context_initial,
                    right_context_final=right_context_final,
                    left_tolerance=(left_tolerance
                                    if left_tolerance is not None
                                    else ''),
                    right_tolerance=(right_tolerance
                                     if right_tolerance is not None
                                     else ''),
                    frame_subsampling_factor=frame_subsampling_factor,
                    alignment_subsampling_factor=alignment_subsampling_factor,
                    stage=stage, frames_per_iter=frames_per_iter,
                    frames_per_eg_str=frames_per_eg_str, srand=srand,
                    deriv_weights_scp=(deriv_weights_scp
                                       if deriv_weights_scp is not None
                                       else ''),
                    data=data, lat_dir=lat_dir, dir=dir, egs_dir=egs_dir,
                    egs_opts=egs_opts if egs_opts is not None else ''))

def train_one_iteration_adapt(dir, iter, srand, egs_dir,
                        num_jobs, num_archives_processed, num_archives,
						learning_rate,
                        num_chunk_per_minibatch_str,
                        apply_deriv_weights, min_deriv_time,
                        max_deriv_time_relative,
                        l2_regularize, xent_regularize,
                        leaky_hmm_coefficient,
                        momentum, max_param_change, shuffle_buffer_size,
                        frame_subsampling_factor,
                        run_opts, dropout_edit_string="", temperature_edit_string="", train_opts="",
                        backstitch_training_scale=0.0, backstitch_training_interval=1,
                        use_multitask_egs=False):
    """ Called from steps/nnet3/chain/train.py for one iteration for
    neural network training with LF-MMI objective

    """

    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    # check if different iterations use the same random seed
    if os.path.exists('{0}/srand'.format(dir)):
        try:
            saved_srand = int(open('{0}/srand'.format(dir)).readline().strip())
        except (IOError, ValueError):
            logger.error("Exception while reading the random seed "
                         "for training")
            raise
        if srand != saved_srand:
            logger.warning("The random seed provided to this iteration "
                           "(srand={0}) is different from the one saved last "
                           "time (srand={1}). Using srand={0}.".format(
                               srand, saved_srand))
    else:
        with open('{0}/srand'.format(dir), 'w') as f:
            f.write(str(srand))

    # Sets off some background jobs to compute train and
    # validation set objectives
    chain_lib.compute_train_cv_probabilities(
        dir=dir, iter=iter, egs_dir=egs_dir,
        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
        leaky_hmm_coefficient=leaky_hmm_coefficient, run_opts=run_opts,
        use_multitask_egs=use_multitask_egs)

    if iter > 0:
        # Runs in the background
        chain_lib.compute_progress(dir, iter, run_opts)

    do_average = (iter > 0)

    raw_model_string = ("nnet3-am-copy --raw=true --learning-rate={0} "
                        "{1}/{2}.mdl - |".format(
                           learning_rate, dir, iter))

    if do_average:
        cur_num_chunk_per_minibatch_str = num_chunk_per_minibatch_str
        cur_max_param_change = max_param_change
    else:
        # on iteration zero, use a smaller minibatch size (and we will later
        # choose the output of just one of the jobs): the model-averaging isn't
        # always helpful when the model is changing too fast (i.e. it can worsen
        # the objective function), and the smaller minibatch size will help to
        # keep the update stable.
        cur_num_chunk_per_minibatch_str = common_train_lib.halve_minibatch_size_str(
            num_chunk_per_minibatch_str)
        cur_max_param_change = float(max_param_change) / math.sqrt(2)

    raw_model_string = raw_model_string + dropout_edit_string + temperature_edit_string
    chain_lib.train_new_models(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
                     num_archives_processed=num_archives_processed,
                     num_archives=num_archives,
                     raw_model_string=raw_model_string,
                     egs_dir=egs_dir,
                     apply_deriv_weights=apply_deriv_weights,
                     min_deriv_time=min_deriv_time,
                     max_deriv_time_relative=max_deriv_time_relative,
                     l2_regularize=l2_regularize,
                     xent_regularize=xent_regularize,
                     leaky_hmm_coefficient=leaky_hmm_coefficient,
                     momentum=momentum,
                     max_param_change=cur_max_param_change,
                     shuffle_buffer_size=shuffle_buffer_size,
                     num_chunk_per_minibatch_str=cur_num_chunk_per_minibatch_str,
                     frame_subsampling_factor=frame_subsampling_factor,
                     run_opts=run_opts, train_opts=train_opts,
                     # linearly increase backstitch_training_scale during the
                     # first few iterations (hard-coded as 15)
                     backstitch_training_scale=(backstitch_training_scale *
                         iter / 15 if iter < 15 else backstitch_training_scale),
                     backstitch_training_interval=backstitch_training_interval,
                     use_multitask_egs=use_multitask_egs)

    [models_to_average, best_model] = common_train_lib.get_successful_models(
         num_jobs, '{0}/log/train.{1}.%.log'.format(dir, iter))
    nnets_list = []
    for n in models_to_average:
        nnets_list.append("{0}/{1}.{2}.raw".format(dir, iter + 1, n))

    if do_average:
        # average the output of the different jobs.
        common_train_lib.get_average_nnet_model(
            dir=dir, iter=iter,
            nnets_list=" ".join(nnets_list),
            run_opts=run_opts)

    else:
        # choose the best model from different jobs
        common_train_lib.get_best_nnet_model(
            dir=dir, iter=iter,
            best_model_index=best_model,
            run_opts=run_opts)

    try:
        for i in range(1, num_jobs + 1):
            os.remove("{0}/{1}.{2}.raw".format(dir, iter + 1, i))
    except OSError:
        raise Exception("Error while trying to delete the raw models")

    new_model = "{0}/{1}.mdl".format(dir, iter + 1)

    if not os.path.isfile(new_model):
        raise Exception("Could not find {0}, at the end of "
                        "iteration {1}".format(new_model, iter))
    elif os.stat(new_model).st_size == 0:
        raise Exception("{0} has size 0. Something went wrong in "
                        "iteration {1}".format(new_model, iter))
    if os.path.exists("{0}/cache.{1}".format(dir, iter)):
        os.remove("{0}/cache.{1}".format(dir, iter))

