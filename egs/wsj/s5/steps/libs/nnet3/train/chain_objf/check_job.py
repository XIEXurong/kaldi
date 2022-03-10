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
import subprocess, time

dir = sys.argv[1]
job = int(sys.argv[2])

print("Checking job: "+dir+"/tmp_parallel/."+str(job))

while True:
    check =  subprocess.check_output("cat {dir}/tmp_parallel/.{job}".format(dir=dir, job=job), shell=True)
    check = int(check)
#    print("Checking job: "+dir+"/tmp_parallel/."+str(job)+", with label: "+str(check))
    if check == 1:
        subprocess.call("rm {dir}/tmp_parallel/.{job}".format(dir=dir, job=job), shell=True)
        break
    time.sleep(5)
