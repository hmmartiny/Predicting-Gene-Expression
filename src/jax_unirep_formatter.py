#!/usr/bin/python3

import argparse
import pandas as pd
from jax_unirep import get_reps


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
        '-a', '--annotations',
        type=str,
        help='File with solubility annotations for sequences in the fasta file (optional)',
        dest='solubility_annotations',
        default=None
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Name to store output dataframe in',
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

def convert(sequences):
    """Format sequences with jax-UniRep"""
    h_avg, _, _ = get_reps(sequences)
    return h_avg


if __name__ == "__main__":
    args = parse_args()

    # load sequences from FASTA file
    seqs = read_fasta(args.fasta)

    # Convert dict to a dataframe and do some formatting
    df = pd.DataFrame.from_dict(seqs, orient='index').reset_index()
    df.rename(columns={'index': 'sid', 0: 'sequence'}, inplace=True)
    df['sid'] = df['sid'].astype('int')

    # If annotations are given, add them to the dataframe.
    if args.solubility_annotations is not None:
        solubility = pd.read_csv(args.solubility_annotations)

        df = df.merge(solubility, on='sid')

    # run unirep formatting
    df['unirep'] = df['sequence'].apply(convert)

    # write df to csv file
    df.to_csv(args.output_file, index=None)