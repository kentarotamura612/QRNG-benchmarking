#!/Users/kentaro/anaconda3/bin/python3

import argparse
import pickle
import sys
import glob
import os
import pandas as pd
import numpy as np
from scipy import special
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

def get_args():
    parser = argparse.ArgumentParser(description="convert sequences into a single text file")
    parser.add_argument("-d", "--dir", help="the name of directory to store files", default=None)
    #parser.add_argument("-e", "--e", help="level of significance for autocorr test", type = np.float32, default = 0.01)
    return parser.parse_args()

def get_curr_dir():
    import os
    return os.getcwd().rstrip("/")

def create_dir(dir):
    import os
    try:
        if not os.path.exists(dir):
            os.mkdir(dir)
        if not os.path.exists(dir+"/autocorr"):
            os.mkdir(dir+"/autocorr")
        if not os.path.exists(dir+"/figures"):
            os.mkdir(dir+"/figures")
    except FileExistsError:
        pass

def listup_files(dir):
    return [os.path.abspath(p) for p in glob.glob(dir+"/*")]

def pro(seq):
    return np.sum(seq)/len(seq)

def xor(a,b):
    return np.bitwise_xor(a, b)

def auto(a, pro_a, l=1):
    import math
    from scipy import special
    b = [a[k] for k in range(l,len(a))]
    a = [a[k] for k in range(0,len(a)-l)]
    sample_size = len(a)
    SUM = sum([xor(a[k],b[k]) for k in range(sample_size)])
    pro = pro_a + pro_a - 2*pro_a*pro_a
    obs = (SUM-pro*sample_size)/math.sqrt(sample_size*pro*(1-pro))
    pval = special.erfc(abs(obs)/math.sqrt(2))
    return obs, pval

def convert_and_filter(pvals, blackth = 0.01, whiteth = 0.1):
    """
    this is a function to filter values of data to be plotted
    value below blackth is black, above whiteth is white

    """
    cols, rows = pvals.shape
    Z = np.zeros((rows,cols,3))
    #set RGV values
    for i in range(rows):
        for j in range(cols):
            if pvals[j][i] < blackth:
                Z[i,j] = [0, 0, 0]            #black
            elif pvals[j][i] >= whiteth:
                Z[i,j] = [1, 1, 1]            #white
            else:
                Z[i,j] = [0.5, 0.5, 0.5]
    return Z

def main(args):
    NQBITS = len(listup_files(args.dir+"/text"))
    NSAMPS = len(listup_files(args.dir+"/sequence"))

    pvals = np.ones((NQBITS, NSAMPS), dtype = np.float32)
    observed = np.ones((NQBITS, NSAMPS), dtype = np.float32)
    for qnum in range(NQBITS):
        f = open(f"{args.dir}/text/qubit{qnum}.txt", 'r')
        sequence = list(map(int,f.read()))
        NSHOTS = len(sequence)//NSAMPS
        for index in range(NSAMPS):
            print(f"checking temporal correlation of qubit{qnum} sample{index+1}")
            seq = sequence[index*NSHOTS:index*NSHOTS+NSHOTS]
            prop = pro(seq)
            obs, pval = auto(seq, prop, 1)
            pvals[qnum][index] = pval
            observed[qnum][index] = obs
        f.close()
    np.savetxt(f"{args.dir}/autocorr/1_pvals.txt", pvals)
    np.savetxt(f"{args.dir}/autocorr/2_obs.txt", observed)

    print("creating temporal correlation figure")

    size = 18
    blackth = 0.01
    whiteth = 0.1
    xlim, ylim = pvals.shape
    plt.figure(num=None, figsize=(15, 20), dpi=100, facecolor='w', edgecolor='k')
    plt.imshow(convert_and_filter(pvals, blackth, whiteth), aspect="auto", interpolation="nearest")
    #plt.imshow(df, aspect="auto", interpolation="nearest")
    plt.xticks(np.arange(xlim, step=1), fontsize=size)
    plt.yticks(np.arange(ylim, step=10), fontsize=size)
    plt.ylabel("Sample number", fontsize=size)
    plt.xlabel("Qubit number", fontsize=size)

    colors = [ [0,0,0], [0.5,0.5,0.5], [1,1,1] ]
    labels = [ "p-value < %.2f" % blackth, " %.2f" % blackth +  r"$\leq$" + "p-value < %.2f" % whiteth,  "%.2f" % whiteth +  "$\leq$" + "p-value" ]
    # create a patch (proxy artist) for every color
    patches = [ mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)  ]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=3, mode="expand", borderaxespad=0., shadow=False, fontsize=size)

    pdf = PdfPages(args.dir+"/figures/temporal_correlation_bitmap.pdf")
    pdf.savefig()
    pdf.close()
    plt.close()

if __name__ == "__main__":
        args = get_args()
        if args.dir is None:
            args.dir = get_curr_dir()
            create_dir(args.dir)
        else:
            args.dir = get_curr_dir() + "/" + args.dir
            create_dir(args.dir)
        main(args)
        exit("done")
