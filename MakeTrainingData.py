import numpy as np
import gzip
import os
import sys

def load_npygz(filepath):
    with gzip.GzipFile(filepath, "rb") as f:
        a = np.load(f)
    return a




chr_list=[]
with open("/home/cog/smehrem/MinorResearchInternship/BAM/BAM_chr_list", "r") as f:
    for line in f:
        line = line.strip()
        chr_list += [line]

chr_list = chr_list[3::]
callers = ["delly", "gridss", "manta", "lumpy"]


labels = []
bin_labels = []
labels_neg=[]
bin_labels_neg=[]
windowpairs = []
windowpair_ids = []
windowpairs_neg = []
windowpair_ids_neg = []

for chrom in chr_list:
    #perchr_callers = []
    for caller in callers:
        filepath = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/"+caller+"/"+chrom+"_windowpairs_DEL.npy.gz"
        filepath_id = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/" + caller + "/" + chrom + "_windowpairs_DEL_winids.npy.gz"
        if os.path.isfile(filepath):
            a = load_npygz(filepath)
            windowpairs.append(a)
            b = load_npygz(filepath_id)
            windowpair_ids.append(b)
            lab = ["DEL"]*a.shape[0]
            bin = [1]*len(b)
            labels += lab
            bin_labels += bin
            #perchr_callers.append(load_npygz(filepath))
   #perchr_callers = np.vstack(perchr_callers)

    filepath_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_NoBPNoBP/"+chrom+"_negative_windowpairs_DEL.npy.gz"
    filepath_id_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_NoBPNoBP/"+chrom+"_negative_windowpairs_DEL_winids.npy.gz"
    if os.path.isfile(filepath_neg):
        c = load_npygz(filepath_neg)
        windowpairs_neg.append(c)
        d = load_npygz(filepath_id_neg)
        windowpair_ids_neg.append(d)
        lab_neg=["No_DEL"]*c.shape[0]
        bin_neg=[0]*len(d)
        labels_neg += lab_neg
        bin_labels_neg += bin_neg




windowpairs = np.vstack(windowpairs)
windowpair_ids = np.concatenate(windowpair_ids)


print("\t".join([str(windowpairs.shape[0]), str(windowpair_ids.shape[0]), str(len(labels)), str(len(bin_labels))]))

windowpairs_neg = np.vstack(windowpairs_neg)
windowpair_ids_neg = np.concatenate(windowpair_ids_neg)

print("\t".join([str(windowpairs_neg.shape[0]), str(windowpair_ids_neg.shape[0]), str(len(labels_neg)), str(len(bin_labels_neg))]))


windowpairs_all = np.vstack([windowpairs, windowpairs_neg])
ids_all = np.concatenate([windowpair_ids, windowpair_ids_neg])
labels_all = np.concatenate([labels, labels_neg])
bin_all = np.concatenate([bin_labels, bin_labels_neg])

print("\t".join([str(windowpairs_all.shape[0]), str(ids_all.shape[0]), str(labels_all.shape[0]), str(bin_all.shape[0])]))

np.savez("N12878_DEL_TrainingData", X=windowpairs_all, y=labels_all, y_binary=bin_all, ids=ids_all)