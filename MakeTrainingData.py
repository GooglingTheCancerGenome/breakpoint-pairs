import numpy as np
import gzip
import os
from keras.utils.np_utils import to_categorical

def load_npygz(filepath):
    with gzip.GzipFile(filepath, "rb") as f:
        a = np.load(f)
    return a

def stack_ontop(window_array):
    new_windows = []
    for pair in window_array:
        sliced1 = pair[0:200, ::]
        sliced2 = pair[210::, ::]
        new_windows += [np.concatenate([sliced1, sliced2], axis=1)]
    new_windows = np.asarray(new_windows)
    return new_windows

channels = []
with open("Channel_Labels.txt","r") as inlabels:
    for line in inlabels:
        line = line.strip()
        channels += [line]

windows_ontop = False
shuffle = True

chr_list = []
with open("/home/cog/smehrem/MinorResearchInternship/BAM/BAM_chr_list", "r") as f:
    for line in f:
        line = line.strip()
        chr_list += [line]

chr_list = chr_list[3:-2]

callers = ["delly", "gridss", "manta", "lumpy" ]


for caller in ["delly"]:
    labels = []
    bin_labels = []
    labels_neg = []
    bin_labels_neg = []
    windowpairs = []
    windowpair_ids = []
    windowpairs_neg = []
    windowpair_ids_neg = []
    for chrom in chr_list:
        perchr = []
        filepath = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/"+caller+"/"+chrom+"_windowpairs_DEL.npy.gz"
        filepath_id = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/" + caller + "/" + chrom + "_windowpairs_DEL_winids.npy.gz"
        if os.path.isfile(filepath):
            a = load_npygz(filepath)
            windowpairs.append(a)
            b = load_npygz(filepath_id)
            windowpair_ids.append(b)
            lab = ["DEL"]*a.shape[0]
            labels += lab
            perchr = load_npygz(filepath)



        filepath_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_TrueSV_TrueSV/"+chrom+"_negative_windowpairs_DEL.npy.gz"
        filepath_id_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_TrueSV_TrueSV/"+chrom+"_negative_windowpairs_DEL_winids.npy.gz"
        if os.path.isfile(filepath_neg) and os.path.isfile(filepath):
            c = load_npygz(filepath_neg)
            e = c[0:perchr.shape[0], :, :]
            windowpairs_neg.append(e)
            d = load_npygz(filepath_id_neg)
            f = d[0:perchr.shape[0]]
            windowpair_ids_neg.append(f)
            lab_neg=["No_DEL"]*f.shape[0]
            labels_neg += lab_neg

    test_set_windowpairs = []
    test_set_ids = []
    test_set_labels = []
    test_set_labels_bin = []
    test_set_windowpairs_neg = []
    test_set_ids_neg = []
    test_set_labels_neg = []
    test_set_labels_bin_neg = []

    for chrom in ["1", "2", "3"]:
        testperchr = []
        filepath = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/" + caller + "/" + chrom + "_windowpairs_DEL.npy.gz"
        filepath_id = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs/" + caller + "/" + chrom + "_windowpairs_DEL_winids.npy.gz"
        if os.path.isfile(filepath):
            g = load_npygz(filepath)
            test_set_windowpairs.append(g)
            h = load_npygz(filepath_id)
            test_set_ids.append(h)
            lab_test = ["DEL"] * g.shape[0]
            test_set_labels += lab_test
            testperchr = load_npygz(filepath)
        filepath_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_TrueSV_TrueSV/" + chrom + "_negative_windowpairs_DEL.npy.gz"
        filepath_id_neg = "/home/cog/smehrem/breakpoint-pairs/genomewide_windowpairs_TrueSV_TrueSV/" + chrom + "_negative_windowpairs_DEL_winids.npy.gz"
        if os.path.isfile(filepath_neg) and os.path.isfile(filepath):
            j = load_npygz(filepath_neg)
            k = j[0:testperchr.shape[0], :, :]
            test_set_windowpairs_neg.append(k)
            n = load_npygz(filepath_id_neg)
            m = n[0:testperchr.shape[0]]
            test_set_ids_neg.append(m)
            test_lab_neg = ["No_DEL"] * k.shape[0]
            test_set_labels_neg += test_lab_neg


    windowpairs = np.vstack(windowpairs)
    windowpairs_neg = np.vstack(windowpairs_neg)
    windowpairs_all = np.vstack([windowpairs, windowpairs_neg])
    windowpair_ids = np.concatenate(windowpair_ids)
    windowpair_ids_neg = np.concatenate(windowpair_ids_neg)
    ids_all = np.concatenate([windowpair_ids, windowpair_ids_neg])
    labels_all = labels + labels_neg
    labels_all = np.asarray(labels_all)


    test_set_windowpairs = np.vstack(test_set_windowpairs)
    test_set_windowpairs_neg = np.vstack(test_set_windowpairs_neg)
    test_set_windowpairs_all = np.vstack([test_set_windowpairs, test_set_windowpairs_neg])
    shuffle_dict = {}
    shuffle_dict_neg = {}

    if shuffle:
        for i in range(0, test_set_windowpairs_all.shape[2]):
            test_set_windowpairs_all_shuffled = np.array(test_set_windowpairs_all, copy=True)
            for j in range(0, test_set_windowpairs_all.shape[0]):
                test_set_windowpairs_all_shuffled[j, :, i] = np.random.permutation(test_set_windowpairs_all_shuffled[j, :, i])
                #print(test_set_windowpairs_all[j, :, i])
                #print(test_set_windowpairs_all_shuffled[j, :, i])
            shuffle_dict[i] = test_set_windowpairs_all_shuffled





    test_set_ids = np.concatenate(test_set_ids)
    test_set_ids_neg = np.concatenate(test_set_ids_neg)
    test_set_ids_all = np.concatenate([test_set_ids, test_set_ids_neg])
    test_set_labels_all = test_set_labels + test_set_labels_neg
    test_set_labels_all = np.asarray(test_set_labels_all)


    classlabels = sorted(list(set(labels_all)))
    mapclasses = {classlabels[i] : i for i in range(len(classlabels))}
    y_train = np.array([mapclasses[c] for c in labels_all], dtype='int')
    y_test = np.array([mapclasses[c] for c in test_set_labels_all], dtype='int')
    y_train_binary = to_categorical(y_train)
    y_test_binary = to_categorical(y_test)

    #if windows_ontop:
        #windowpairs_all = stack_ontop(windowpairs_all)
        #test_set_windowpairs_all = stack_ontop(test_set_windowpairs_all)
       # np.savez("N12878_DEL_TrainingData_"+caller+"_stacked", X=windowpairs_all, y=labels_all, y_binary=y_train_binary, ids=ids_all)
       # np.savez("N12878_DEL_TestData_"+caller+"_stacked", X=test_set_windowpairs_all, y=test_set_labels_all, y_binary=y_test_binary, ids=test_set_ids_all)
    #else:
        #np.savez("Test_Training_Data/CR_TrueSV/N12878_DEL_TrainingData_"+caller, X=windowpairs_all, y=labels_all, y_binary=y_train_binary, ids=ids_all)
        #np.savez("Test_Training_Data/CR_TrueSV/N12878_DEL_TestData_"+caller, X=test_set_windowpairs_all, y=test_set_labels_all, y_binary=y_test_binary, ids=test_set_ids_all)

    if shuffle:
        for key in shuffle_dict:
            shuffled_key = shuffle_dict[key]
            np.savez("Test_Training_Data/shuffled/TrueSV_TrueSV/N12878_DEL_TestData_"+channels[key]+"_"+caller, X=shuffled_key, y=test_set_labels_all, y_binary=y_test_binary, ids=test_set_ids_all)
            print(key)
            print(caller)
