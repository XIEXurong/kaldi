#!/usr/bin/env python3

import argparse
import sys
import numpy as np

parser = argparse.ArgumentParser(description="Compute similarity between features.")
parser.add_argument('--similar_score', type=str, default='COS',
                    help='type of similarity score. can be COS, PROD or DIST.')
parser.add_argument('base_feat', type=str,
                    help='File of base features.')
parser.add_argument('target_feat', type=str,
                    help='File of target features.')
parser.add_argument('select_name', type=str,
                    help='File to store the selected tokens.')
parser.add_argument('select_number', type=int,
                    help='Number of tokens to be selected.')

args = parser.parse_args()
     
base_feat = args.base_feat # sys.argv[1]
target_feat = args.target_feat # sys.argv[2]
select_name = args.select_name # sys.argv[3]
select_number = args.select_number # sys.argv[4]

base_id = []
target_id = []
base_mat = []
target_mat = []

f = open(base_feat, 'r', encoding='utf-8')
print("Reading pool of base features from", base_feat)

for line in f:
    feats = line.split()
    base_id.append(feats[0])
    base_mat.append(feats[2:-1])

f.close()

base_mat = np.array(base_mat,dtype='float32')

f = open(target_feat, 'r', encoding='utf-8')
print("Reading target features from", target_feat)

for line in f:
    feats = line.split()
    target_id.append(feats[0])
    target_mat.append(feats[2:-1])

f.close()

target_mat = np.array(target_mat,dtype='float32')

if args.similar_score == 'COS':
    print("Computing COS scores and selecting the first", select_number, "tokens")
    base_mat_norm = (np.sum(base_mat**2, axis=1)**0.5).reshape(-1,1)
    target_mat_norm = (np.sum(target_mat**2, axis=1)**0.5).reshape(-1,1)
    score = -1 * np.matmul(base_mat,target_mat.T)/np.matmul(base_mat_norm,target_mat_norm.T)
elif args.similar_score == 'PROD':
    print("Computing PROD scores and selecting the first", select_number, "tokens")
    score = -1 * np.matmul(base_mat,target_mat.T)
elif args.similar_score == 'DIST':
    print("Computing DIST scores and selecting the first", select_number, "tokens")
    base_mat_norm2 = (np.sum(base_mat**2, axis=1)).reshape(-1,1)
    target_mat_norm2 = (np.sum(target_mat**2, axis=1)).reshape(-1,1)
    score = base_mat_norm2 + target_mat_norm2.T - 2 * np.matmul(base_mat,target_mat.T)
else:
    raise ValueError("""An invalid option for `--similar_score` was supplied, options are ['COS', 'PROD' or 'DIST']""")

sort_id = np.argsort(score, axis=0, kind='mergesort')
select_id = sort_id[:int(select_number)]


f = open(select_name, 'w', encoding='utf-8')
print("Writting selected tokens to", select_name)

for i in range(len(target_id)):
    key = target_id[i]
    r = f.write(key)
    for j in range(int(select_number)):
        select_key = base_id[select_id[j][i]]
        r = f.write(" "+select_key)
    r = f.write("\n")

f.close()

