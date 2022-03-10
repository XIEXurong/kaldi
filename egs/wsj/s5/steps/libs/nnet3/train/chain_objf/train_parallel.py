from __future__ import division
from __future__ import print_function

import logging
import math
import os
import sys

import libs.common as common_lib
import libs.nnet3.train.common as common_train_lib
from libs.nnet3.train.chain_objf.acoustic_model import *

def train_new_models_parallel(dir, iter, srand, num_jobs,
                     num_archives_processed, num_archives,
                     raw_model_string, egs_dir,
                     apply_deriv_weights,
                     min_deriv_time, max_deriv_time_relative,
                     l2_regularize, xent_regularize, leaky_hmm_coefficient,
                     momentum, max_param_change,
                     shuffle_buffer_size, num_chunk_per_minibatch_str,
                     frame_subsampling_factor, run_opts, train_opts, free_memory_required,
                     backstitch_training_scale=0.0, backstitch_training_interval=1,
                     use_multitask_egs=False):
    """
    Called from train_one_iteration(), this method trains new models
    with 'num_jobs' jobs, and
    writes files like exp/tdnn_a/24.{1,2,3,..<num_jobs>}.raw

    We cannot easily use a single parallel SGE job to do the main training,
    because the computation of which archive and which --frame option
    to use for each job is a little complex, so we spawn each one separately.
    this is no longer true for RNNs as we use do not use the --frame option
    but we use the same script for consistency with FF-DNN code

    use_multitask_egs : True, if different examples used to train multiple
                        tasks or outputs, e.g.multilingual training.
                        multilingual egs can be generated using get_egs.sh and
                        steps/nnet3/multilingual/allocate_multilingual_examples.py,
                        those are the top-level scripts.
    """

    deriv_time_opts = []
    if min_deriv_time is not None:
        deriv_time_opts.append("--optimization.min-deriv-time={0}".format(
                                    min_deriv_time))
    if max_deriv_time_relative is not None:
        deriv_time_opts.append("--optimization.max-deriv-time-relative={0}".format(
                                    int(max_deriv_time_relative)))

    

    threads = []
    # the GPU timing info is only printed if we use the --verbose=1 flag; this
    # slows down the computation slightly, so don't accumulate it on every
    # iteration.  Don't do it on iteration 0 either, because we use a smaller
    # than normal minibatch size, and people may get confused thinking it's
    # slower for iteration 0 because of the verbose option.
    verbose_opt = ("--verbose=1" if iter % 20 == 0 and iter > 0 else "")

    thread = common_lib.background_command("""python3 steps/libs/nnet3/train/chain_objf/check_free_memory_gpu.py {dir} {num_jobs} {free_memory_required}""".format(dir=dir, num_jobs=num_jobs, free_memory_required=free_memory_required), require_zero_status=True)
    threads.append(thread)

    import subprocess, time
    while not os.path.isfile(dir+"/tmp_parallel/on"):
        time.sleep(5)
    common_lib.execute_command("""rm {dir}/tmp_parallel/on""".format(dir=dir))
    
    for job in range(1, num_jobs+1):
        if os.path.isfile(dir+"/"+str(iter+1)+"."+str(job)+".raw"):
            subprocess.call("rm {dir}/tmp_parallel/.{job}".format(dir=dir, job=job), shell=True)
            continue # if .raw exist, then no need to execute this job
        # k is a zero-based index that we will derive the other indexes from.
        k = num_archives_processed + job - 1
        # work out the 1-based archive index.
        archive_index = (k % num_archives) + 1
        # previous : frame_shift = (k/num_archives) % frame_subsampling_factor
        frame_shift = ((archive_index + k//num_archives)
                       % frame_subsampling_factor)

        multitask_egs_opts = common_train_lib.get_multitask_egs_opts(
            egs_dir,
            egs_prefix="cegs.",
            archive_index=archive_index,
            use_multitask_egs=use_multitask_egs)
        scp_or_ark = "scp" if use_multitask_egs else "ark"
        cache_io_opts = (("--read-cache={dir}/cache.{iter}".format(dir=dir,
                                                                  iter=iter)
                          if iter > 0 else "") +
                         (" --write-cache={0}/cache.{1}".format(dir, iter + 1)
                          if job == 1 else ""))

        thread = common_lib.background_command(
            """python3 steps/libs/nnet3/train/chain_objf/check_job.py {dir} {job}
            {command} {train_queue_opt} {dir}/log/train.{iter}.{job}.log \
                    nnet3-chain-train {parallel_train_opts} {verbose_opt} \
                    --apply-deriv-weights={app_deriv_wts} \
                    --l2-regularize={l2} --leaky-hmm-coefficient={leaky} \
                    {cache_io_opts}  --xent-regularize={xent_reg} \
                    {deriv_time_opts} \
                    --print-interval=10 --momentum={momentum} \
                    --max-param-change={max_param_change} \
                    --backstitch-training-scale={backstitch_training_scale} \
                    --backstitch-training-interval={backstitch_training_interval} \
                    --l2-regularize-factor={l2_regularize_factor} {train_opts} \
                    --srand={srand} \
                    "{raw_model}" {dir}/den.fst \
                    "ark,bg:nnet3-chain-copy-egs {multitask_egs_opts} \
                        --frame-shift={fr_shft} \
                        {scp_or_ark}:{egs_dir}/cegs.{archive_index}.{scp_or_ark} ark:- | \
                        nnet3-chain-shuffle-egs --buffer-size={buf_size} \
                        --srand={srand} ark:- ark:- | nnet3-chain-merge-egs \
                        --minibatch-size={num_chunk_per_mb} ark:- ark:- |" \
                    {dir}/{next_iter}.{job}.raw""".format(
                        command=run_opts.command,
                        train_queue_opt=run_opts.train_queue_opt,
                        dir=dir, iter=iter, srand=iter + srand,
                        next_iter=iter + 1, job=job,
                        deriv_time_opts=" ".join(deriv_time_opts),
                        app_deriv_wts=apply_deriv_weights,
                        fr_shft=frame_shift, l2=l2_regularize,
                        train_opts=train_opts,
                        xent_reg=xent_regularize, leaky=leaky_hmm_coefficient,
                        cache_io_opts=cache_io_opts,
                        parallel_train_opts=run_opts.parallel_train_opts,
                        verbose_opt=verbose_opt,
                        momentum=momentum, max_param_change=max_param_change,
                        backstitch_training_scale=backstitch_training_scale,
                        backstitch_training_interval=backstitch_training_interval,
                        l2_regularize_factor=1.0/num_jobs,
                        raw_model=raw_model_string,
                        egs_dir=egs_dir, archive_index=archive_index,
                        buf_size=shuffle_buffer_size,
                        num_chunk_per_mb=num_chunk_per_minibatch_str,
                        multitask_egs_opts=multitask_egs_opts,
                        scp_or_ark=scp_or_ark),
            require_zero_status=True)

        threads.append(thread)

    for thread in threads:
        thread.join()


def train_one_iteration_parallel(dir, iter, srand, egs_dir,
                        num_jobs, num_archives_processed, num_archives,
                        learning_rate, shrinkage_value,
                        num_chunk_per_minibatch_str,
                        apply_deriv_weights, min_deriv_time,
                        max_deriv_time_relative,
                        l2_regularize, xent_regularize,
                        leaky_hmm_coefficient,
                        momentum, max_param_change, shuffle_buffer_size,
                        frame_subsampling_factor, free_memory_required,
                        run_opts, dropout_edit_string="", train_opts="",
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
    compute_train_cv_probabilities(
        dir=dir, iter=iter, egs_dir=egs_dir,
        l2_regularize=l2_regularize, xent_regularize=xent_regularize,
        leaky_hmm_coefficient=leaky_hmm_coefficient, run_opts=run_opts,
        use_multitask_egs=use_multitask_egs)

    if iter > 0:
        # Runs in the background
        compute_progress(dir, iter, run_opts)

    do_average = (iter > 0)

    raw_model_string = ("nnet3-am-copy --raw=true --learning-rate={0} "
                        "--scale={1} {2}/{3}.mdl - |".format(
                            learning_rate, shrinkage_value, dir, iter))

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

    raw_model_string = raw_model_string + dropout_edit_string
    train_new_models_parallel(dir=dir, iter=iter, srand=srand, num_jobs=num_jobs,
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
                     run_opts=run_opts, train_opts=train_opts, free_memory_required=free_memory_required,
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


