#!/usr/bin/env python

from __future__ import division
from __future__ import print_function

import argparse
import logging
import os
import pprint
import shutil
import sys
import traceback

sys.path.insert(0, 'steps')
import libs.common as common_lib
import glob, time, re, subprocess

dir = sys.argv[1]
num_jobs = int(sys.argv[2])
free_memory_required = int(sys.argv[3])

os.system("""rm -r {dir}/tmp_parallel""".format(dir=dir))
common_lib.execute_command("""mkdir -p {dir}/tmp_parallel""".format(dir=dir))
for job in range(1, num_jobs+1):
    common_lib.execute_command("""echo 0 > {dir}/tmp_parallel/.{job}""".format(dir=dir, job=job))

common_lib.execute_command("""> {dir}/tmp_parallel/on""".format(dir=dir))

while len(glob.glob(dir+"/tmp_parallel/.*")) > 0:
    gpu_log =  subprocess.check_output("nvidia-smi", shell=True)
    gpu_log = str(gpu_log)
    res = r"Processes:"
    pos = re.search(res, gpu_log)
    gpu_log = gpu_log[:pos.start()]
    res = r"\d+MiB"
    memory = re.search(res, gpu_log)
    while memory != None:
        start1 = memory.start()
        end1 = memory.end()
        mem1 = int(gpu_log[start1:end1-3])
        gpu_log = gpu_log[end1:]
        memory = re.search(res, gpu_log)
        start2 = memory.start()
        end2 = memory.end()
        mem2 = int(gpu_log[start2:end2-3])
        gpu_log = gpu_log[end2:]
        memory = re.search(res, gpu_log)
        free_memory = mem2 - mem1
#        print("Free memory: "+str(free_memory)+" MiB")
        if free_memory > free_memory_required:
            file = glob.glob(dir+"/tmp_parallel/.*")[0]
            print("Allocate to: "+file+" with free memory: "+str(free_memory)+" MiB")
            common_lib.execute_command("""echo 1 > {file}""".format(file=file))
            break
    time.sleep(10)

common_lib.execute_command("""rm -r {dir}/tmp_parallel""".format(dir=dir))
