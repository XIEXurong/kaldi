#!/usr/bin/env python3

import argparse
import sys

parser = argparse.ArgumentParser(description="Compute allowed lengths.")
parser.add_argument('--min_diff', type=int, default=4,
                    help='Minimized difference.')

args = parser.parse_args()

inputs = sys.stdin

lens = []
for line in inputs:
    line = line.strip()
    lens.append(int(line))

lens.sort()
allowed_len = []

len_start = lens[0]
allowed_len.append(str(len_start))
for i in range(len(lens)):
    if lens[i] - len_start > args.min_diff:
        len_start = lens[i]
        allowed_len.append(str(len_start))

print("\n".join(allowed_len))
