#!/Users/kentaro/anaconda3/bin/python3

import argparse
import pickle
import sys
import glob
import os
import pandas as pd
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description="convert sequences into a single text file")
    parser.add_argument("-d", "--dir", help="the name of directory to store files", default=None)
    return parser.parse_args()

def get_curr_dir():
    import os
    return os.getcwd().rstrip("/")


def create_dir(dir):
    import os
    try:
        os.mkdir(dir)
        os.mkdir(dir+"/text")
    except FileExistsError:
        pass


def listup_files(dir):
    return [os.path.abspath(p) for p in glob.glob(dir+"/*")]

def join_list(lst):
    joined = ""
    for element in lst:
        joined += str(element)
    return joined

def main(args):
    paths = pd.read_pickle(listup_files(args.dir+"/paths")[0])
    for path in paths:
        print(f"converting {path} to text files per qubit")
        #index = path.find("/sequence/")
        #jobid = path[index+10:index+10+24]
        sequence = pd.read_pickle(path)
        for qnum in range(len(sequence[0])):
            try:
                with open(f"{args.dir}/text/qubit{qnum}.txt", mode='a') as f:
                    f.writelines([sequence[k][qnum] for k in range(len(sequence))])
            except FileExistsError:
                pass

if __name__ == "__main__":
        args = get_args()
        if args.dir is None:
            args.dir = get_curr_dir()
            if not os.path.exists(args.dir+"/text"):
                os.mkdir(args.dir+"/text")
        else:
            args.dir = get_curr_dir() + "/" + args.dir
            if not os.path.exists(args.dir+"/text"):
                os.mkdir(args.dir+"/text")
        main(args)
        print("Samples successfully converted to text")
