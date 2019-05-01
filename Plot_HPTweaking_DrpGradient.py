import itertools
import matplotlib.pyplot as plt
import statistics
import re
import pandas as pd
import pickle
from operator import itemgetter


f1_dict_drp = {}

callers = ["delly", "lumpy", "manta", "gridss"]

best_models_f1_std = [["30", "100", "5", "delly"], ["20", "50", "3", "lumpy"], ["20", "100", "4", "manta"], ["20", "100", "4", "gridss"]]

drpout_1 = ["10", "20", "30", "40", "50"]
drpout_2 = ["10", "20", "30", "40", "50"]

paras = list(itertools.product(drpout_1, drpout_2))
paras = paras + [("0", "0")]
x = range(0, 10)
f1_per_iter_dict = {}
for par in paras:
    for bm in best_models_f1_std:
        a = bm + [par[0], par[1]]
        f1_per_iter_dict["_".join(a)] = (0,0)
        path = "Results_Hyperparameter_NewBestModels_DropoutGradient_26042019/" + bm[3]+"/NA12878_CNN_results_"+"_".join(a)
        f1_dict_drp["_".join(a)] = []
        for i in range(0, 10):
            with open(path + "/Training_Iteration_" + str(i + 1) + "/NA12878_confusion_matrix_cv_iter_" + str(
                    i + 1) + ".csv", "r") as inconf:
                next(inconf)
                tab = inconf.readlines()
                tab1 = tab[0].strip().split(",")
                tab2 = tab[1].strip().split(",")
                if len(tab1) == 3 and len(tab2) == 3:
                    TP = int(tab1[1])
                    FP = int(tab2[1])
                    TN = int(tab2[2])
                    FN = int(tab1[2])
                else:
                    TP = int(tab1[1])
                    FP = int(tab2[1])
                    TN = 0
                    FN = 0

                recall = TP / (TP + FN)
                precision = TP / (TP + FP)
                f1 = 2 * ((precision * recall) / (precision + recall))
                f1_dict_drp["_".join(a)] += [f1]
                if f1 > f1_per_iter_dict["_".join(a)][0]:
                    f1_per_iter_dict["_".join(a)] = (f1, i)



def make_std_mean(input_tupel):
    input = input_tupel[1]
    std = statistics.stdev(input)
    mean = statistics.mean(input)
    return (mean,std)


fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(16, 10), sharex = True)


for i, caller in enumerate(callers):
    group = []
    for key in f1_dict_drp:
        if caller in key:
            group += [(key, f1_dict_drp[key])]

    #group.sort(key=make_std_mean, reverse = True)
    values = []
    labels = []
    for ele in group:
        values += [ele[1]]
        label = ele[0].split("_")
        labels += ["L1:"+label[-2]+"%, L2:"+label[-1]+"%"]
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    box = ax[i].boxplot(values, labels=labels, positions=range(1, len(group)+1), patch_artist=True, sym='.', medianprops = medianprops)
    ax[i].set_ylim(0, 1)
    ax[i].set_ylabel(caller+"\n F1")
    colors = ['darkkhaki']*len(labels)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.setp(ax[i].get_xticklabels(), rotation=80)
    ax[i].yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)

ax[0].set_title("F1 variability with differing dropout rates for dense layers 1 and 2 per caller\n")
plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
#plt.show()
plt.savefig("figures/F1_variability_all_callers_dropoutGradient_NewBestModel.pdf", format = 'pdf', dpi=300)
plt.close()

best_mean_std_per_caller = {}
for i, caller in enumerate(callers):
    fig, ax = plt.subplots(figsize=(12, 8))
    group = []
    for key in f1_dict_drp:
        if caller in key:
            group += [(key, f1_dict_drp[key])]

    group.sort(key=make_std_mean, reverse = True)
    values = []
    labels = []
    for ele in group:
        values += [ele[1]]
        label = ele[0].split("_")
        labels += ["L1:"+label[-2]+"%, L2:"+label[-1]+"%"]
    best_mean_std_per_caller[caller] = labels[0]
    medianprops = dict(linestyle='-', linewidth=1, color='black')
    box = ax.boxplot(values, labels=labels, positions=range(1, len(group)+1), patch_artist=True, sym='.', medianprops = medianprops)
    ax.set_ylabel("F1")
    colors = ['darkkhaki']*len(labels)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.setp(ax.get_xticklabels(), rotation=80)

    ax.set_title("F1 variability with differing dropout rates for dense layers 1 and 2 for "+caller+".")
    ax.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    #plt.show()
    #plt.savefig("figures/F1_variability_"+caller+"_dropoutGradient_NewBestModel.pdf", format='pdf', dpi=300)
    plt.close()


for best in best_models_f1_std:
    dropoutrates = re.findall(r'(\d*)%', best_mean_std_per_caller[best[3]])
    dropoutrates_p = "_".join(dropoutrates)
    path = "Results_Hyperparameter_NewBestModels_DropoutGradient_26042019/" +best[3]+"/NA12878_CNN_results_"+"_".join(best)+"_"+dropoutrates_p+"/"
    best_model_index = f1_per_iter_dict["_".join(best)+"_"+dropoutrates_p][1]+1
    history = pickle.load(open(
        path + "Training_Iteration_" + str(best_model_index) + "/Best_Model_History_Iteration_" + str(best_model_index),
        "rb"))
    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    x = range(0, len(acc))
    fig, ax1 = plt.subplots()
    ax1.set_title(
        "Best (high mean F1, low std) model of " + best[3] + "\nSplit: " + best[0] + ", epochs: " + best[1] +
        ", learning rate: " + best[2] + ", L1:" + dropoutrates[0] + "%/L2:" + dropoutrates[1] + "%, iteration: " + str(
            best_model_index) + "\n F1:" + str(round(f1_per_iter_dict["_".join(best)+"_"+dropoutrates_p][0], 3)))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    plt.plot(x, acc, color='red', label="Train_Acc")
    plt.plot(x, val_acc, color='lightcoral', label="Val_Acc")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 10)
    plt.plot(x, loss, color='blue', label="Train_Loss")
    plt.plot(x, val_loss, color='lightblue', label="Val_Loss")
    ax2.tick_params(axis='y')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    plt.savefig("figures/Best_highF1_lowSTD_model_of_" + best[3] + "_with_Dropout", format='pdf', dpi=300)

    # plt.show()
    plt.close()

