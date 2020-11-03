#!/usr/bin/python3

import argparse
import pandas as pd
from jax_unirep import get_reps
from tqdm import tqdm
import gc
import os
import pickle

def parse_args():
    parser = argparse.ArgumentParser("Format sequences with jax-UniRep")

    parser.add_argument(
        '-f', '--fasta',
        type=str,
        required=True,
        help="File with fasta sequences to be formatted",
        dest='fasta'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Name to store converted sequences in pickle format',
        dest='output_file'
    )

    return parser.parse_args()

def read_fasta(fastafile):
    """Parse a file with sequences in FASTA format and store in a dict"""
    with open(fastafile, 'r') as f:
        content = [l.strip() for l in f.readlines()]

    res = {}
    seq, seq_id = '', None
    for line in content:
        if line.startswith('>'):
            
            if len(seq) > 0:
                res[seq_id] = seq
            
            seq_id = line.replace('>', '')
            seq = ''
        else:
            seq += line

    return res


def convert(sequence):
    """Format sequences with jax-UniRep"""
    h_avg, _, _ = get_reps(sequence)
    return h_avg

def to_pickle(d, destfile):
    with open(destfile, 'wb') as dest:
        pickle.dump(d, dest)

def load_pickle(sourcefile):
    with open(sourcefile, 'rb') as source:
        d = pickle.load(source)
    return d

if __name__ == "__main__":
    args = parse_args()

    # load sequences from FASTA file
    seqs = read_fasta(args.fasta)

    if os.path.isfile(args.output_file) and os.path.getsize(args.output_file) > 0:
        unirep_seqs = load_pickle(args.output_file)
        for sid in unirep_seqs.keys():
            seqs.pop(sid)
    else:
        unirep_seqs = {}

    for sid, sequence in tqdm(seqs.items()):
        h_avg = convert(sequence)
        unirep_seqs[sid] = h_avg
        
        to_pickle(
            d = unirep_seqs, 
            destfile=args.output_file
        )
        
        del h_avg
        gc.collect()