import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import statistics
import re
import pickle
import numpy as np
import itertools
import pandas as pd

callers = ["delly", "lumpy", "manta", "gridss"]
learning_rates = ["2", "3", "4", "5", "6"]
splits = ["20", "30", "40"]
epochs = ["10", "50", "100"]

f1_dict = {}
mean_dict = {}
std_dict = {}
best_f1_per_caller = {}
best_std_per_caller = {}
best_mean_per_caller = {}

for caller in callers:
    best_f1_per_caller[caller] = []
    best_std = 100
    best_mean = 0
    best_f1 = 0
    for split in splits:
        for epoch in epochs:
            for lr in learning_rates:
                path = "Results_Hyperparameter_Tweaking_16042019/"+caller+"/NA12878_CNN_results_"+split+"_"+epoch+"_"+lr+"_"+caller
                f1_dict["_".join([split, epoch, lr, caller])] = []
                f1_over10 = []
                for i in range(0, 10):
                    with open(path+"/Training_Iteration_"+str(i+1)+"/NA12878_confusion_matrix_cv_iter_"+str(i+1)+".csv","r") as inconf:
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
                        f1 = 2*((precision * recall)/(precision + recall))
                        f1_over10 += [f1]
                        f1_dict["_".join([split, epoch, lr, caller])] += [f1]
                    if f1 > best_f1:
                        best_f1 = f1
                        best_f1_per_caller[caller] = ["_".join([split, epoch, lr, caller]), str(i+1)]

                std = statistics.stdev(f1_over10)
                mean = statistics.mean(f1_over10)
                mean_dict["_".join([split, epoch, lr, caller])] = mean
                std_dict["_".join([split, epoch, lr, caller])] = std
                f1_over10 = np.asarray(f1_over10)
                best_index = np.argmax(f1_over10)
                if std < best_std:
                    best_std = std
                    best_std_per_caller[caller] = ["_".join([split, epoch, lr, caller]), best_std, best_index, f1_over10[best_index]]
                if mean > best_mean:
                    best_mean = mean
                    best_mean_per_caller[caller] = ["_".join([split, epoch, lr, caller]), best_mean, best_index, f1_over10[best_index]]

#print(best_std_per_caller)
#print(best_mean_per_caller)


df_mean = pd.DataFrame.from_dict(mean_dict, orient='index', columns=["Mean_F1"])
df_mean.reset_index(level=0, inplace=True)

df_std = pd.DataFrame.from_dict(std_dict, orient='index', columns=["STD"])
df_std.reset_index(level=0, inplace=True)

df_mean["STD"] = df_mean["index"].map(std_dict)
df_mean[['split', 'epochs', 'learningrate', 'caller']] = df_mean['index'].str.split('_', expand=True)
df_mean = df_mean.sort_values(['Mean_F1', 'STD'], ascending=[False, True])
df_lowstd = df_mean.sort_values(['STD'], ascending=[True])


with open("best_models_highmeanF1_lowstd.tab", "w") as outfile:
    for caller in callers:
        best = df_mean[df_mean['caller'] == caller].iloc[0]
        outfile.write(best.to_string()+'\n\n')

with open("best_models_lowstd.tab", "w") as outfile:
    for caller in callers:
        best = df_lowstd[df_lowstd['caller'] == caller].iloc[0]
        outfile.write(best.to_string()+'\n\n')

x = range(0, 10)
fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(16, 12), sharex=True, sharey=True)
fig.suptitle("F1 variability per caller, learning rate, epochs and split\nover 10 iterations")
for i, caller in enumerate(callers):
    ax[0, i].set_title(caller)
    for key in f1_dict:
        if "2_"+caller in key:
            if "50_2_"+caller in key:
                ax[0, i].plot(x, f1_dict[key], linestyle="dashdot")
                ax[0, i].xaxis.set_major_locator(MaxNLocator(integer=True))
            elif "10_2_"+caller in key:
                ax[0, i].plot(x, f1_dict[key], linestyle="dashed")
            else:
                ax[0, i].plot(x, f1_dict[key])


    for key in f1_dict:
        if "3_"+caller in key:
            if "50_3_"+caller in key:
                ax[1, i].plot(x, f1_dict[key], linestyle="dashdot")
            elif "10_3_"+caller in key:
                ax[1, i].plot(x, f1_dict[key], linestyle="dashed")
            else:
                ax[1, i].plot(x, f1_dict[key])


    for key in f1_dict:
        if "4_"+caller in key:
            key_legend = re.findall(r'((\d*)_(\d*))_\d', key)
            if "50_4_"+caller in key:
                ax[2, i].plot(x, f1_dict[key], linestyle="dashdot", label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            elif "10_4_"+caller in key:
                ax[2, i].plot(x, f1_dict[key], linestyle="dashed", label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            else:
                ax[2, i].plot(x, f1_dict[key], label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])

    for key in f1_dict:
        if "5_"+caller in key:
            key_legend = re.findall(r'((\d*)_(\d*))_\d', key)
            if "50_5_"+caller in key:
                ax[3, i].plot(x, f1_dict[key], linestyle="dashdot", label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            elif "10_5_"+caller in key:
                ax[3, i].plot(x, f1_dict[key], linestyle="dashed", label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            else:
                ax[3, i].plot(x, f1_dict[key], label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])

    for key in f1_dict:
        if "6_"+caller in key:
            key_legend = re.findall(r'((\d*)_(\d*))_\d', key)
            if "50_6_"+caller in key:
                ax[4, i].plot(x, f1_dict[key], linestyle="dashdot", label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            elif "10_6_"+caller in key:
                ax[4, i].plot(x, f1_dict[key], linestyle="dashed",label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])
            else:
                ax[4, i].plot(x, f1_dict[key], label="Split: "+key_legend[0][1]+"%, Epochs: "+key_legend[0][2])

ax[4, 3].legend(loc='center left', bbox_to_anchor = (1, 0.5), ncol=1, prop={'size': 10} )

#pylab.legend(loc=9, bbox_to_anchor=(0.5, -0.1), ncol=2)
fig.text(0.5, 0.04, "Iteration", ha='center')
fig.text(0.04, 0.5, 'F1 score', va='center', rotation='vertical')
ax[0, 0].text(0.01, 0.04, 'Learning rate: 2', ha='left', va='top', rotation='horizontal')
ax[1, 0].text(0.01, 0.04, 'Learning rate: 3', ha='left', va='top', rotation='horizontal')
ax[2, 0].text(0.01, 0.04, 'Learning rate: 4', ha='left', va='top', rotation='horizontal')
ax[3, 0].text(0.01, 0.04, 'Learning rate: 5', ha='left', va='top', rotation='horizontal')
ax[4, 0].text(0.01, 0.04, 'Learning rate: 6', ha='left', va='top', rotation='horizontal')


plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
#plt.show()
#plt.savefig("figures/F1_variability_per_caller.pdf", format = 'pdf', dpi=300)
plt.close()




for best in callers:
    history = pickle.load(open("Results_Hyperparameter_Tweaking_16042019/"+best+"/NA12878_CNN_results_"
                               +best_std_per_caller[best][0]+"/Training_Iteration_" + str(best_std_per_caller[caller][2])
                               + "/Best_Model_History_Iteration_" + str(best_std_per_caller[caller][2]), "rb"))
    results = pd.read_csv("Results_Hyperparameter_Tweaking_16042019/"+best+"/NA12878_CNN_results_"
                               +best_std_per_caller[best][0] + "/CV_results.csv", delimiter="\t", index_col=0)
    best_hyperpar = re.findall(r'(\d*)_(\d*)_(\d)_(\S*)', best_std_per_caller[best][0])
    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    x = range(0, len(acc))
    fig, ax1 = plt.subplots()
    ax1.set_title("Best (low std) model of "+best+"\nSplit: "+best_hyperpar[0][0]+", epochs: "+best_hyperpar[0][1]+
                  ", learning rate: "+best_hyperpar[0][2]+", Iteration: "+str(best_std_per_caller[caller][2]))
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
    #plt.savefig("figures/Best_low_std_model_of_"+best, format='pdf', dpi=300)

    #plt.show()
    plt.close()

f1_dict_drp = {}
best_models_std = [["30", "100", "5", "delly"], ["30", "50", "4", "lumpy"], ["40", "50", "4", "manta"], ["30", "50", "4", "gridss"]]
best_models_f1_std = [["30", "100", "5", "delly"], ["20", "50", "3", "lumpy"], ["20", "100", "4", "manta"], ["20", "100", "4", "gridss"]]

drpout_1 = ["0", "20", "50"]
drpout_2 = ["0", "20", "50"]

paras = itertools.product(drpout_1, drpout_2)
x = range(0, 10)
for par in paras:
    for bm in best_models_f1_std:
        a = bm + [par[0], par[1]]
        path = "Results_Hyperparameter_NewBestModels_Dropout_26042019/"+bm[3]+"/NA12878_CNN_results_"+"_".join(a)
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
#print(f1_dict_drp.keys())

fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(14, 12), sharex=True, sharey=True)
fig.suptitle("F1 variability of best models after adding dropout (high mean F1, low std)")
for i, caller in enumerate(callers):
    for ele in best_models_f1_std:
        if ele[3] == caller:
            title = ele[3]+": "+"_".join(ele[0:3])
    ax[0, i].set_title(title)
    ax[0, i].set_ylim(0.3, 1)
    for key in f1_dict_drp:
        if caller+"_0_" in key:
            #print(key+"Y")

            if caller+"_0_20" in key:
                ax[0, i].plot(x, f1_dict_drp[key], linestyle="dashdot", label="L1:0%, L2:20%")
            elif caller+"_0_50" in key:
                ax[0, i].plot(x, f1_dict_drp[key], linestyle="dashed",  label="L1:0%, L2:50%")
            else:
                ax[0, i].plot(x, f1_dict_drp[key], label = "L1:0%, L2:0%")

    for key in f1_dict_drp:
        ax[1, i].set_ylim(0.3, 1)
        if caller in key and re.search(r'_[a-z]*_[0]_[0]|_[a-z]*_((20|50)_(0))', key):
            #print(key+"X")

            if caller+"_20_0" in key:
                ax[1, i].plot(x, f1_dict_drp[key], linestyle="dashdot",  label="L1:20%, L2:0%")
            elif caller+"_50_0" in key:
                ax[1, i].plot(x, f1_dict_drp[key], linestyle="dashed",  label="L1:50%, L2:0%")
            else:
                ax[1, i].plot(x, f1_dict_drp[key], label = "L1:0%, L2:0%")

    for key in f1_dict_drp:
        ax[2, i].set_ylim(0.3, 1)
        if caller in key and re.search(r'_[a-z]*_[0]_[0]|_[a-z]*_((20|50)_(20|50))', key):
            #print(key+"Z")
            key_legend = re.findall(r'\d*_\d*_\d*_\D*_((\d*)_(\d*))', key)

            if caller+"_20_" in key:
                ax[2, i].plot(x, f1_dict_drp[key], linestyle="dashdot", label="L1:"+key_legend[0][1]+"%, L2:"+key_legend[0][2]+"%")
            elif caller+"_50_" in key:
                ax[2, i].plot(x, f1_dict_drp[key], linestyle="dashed", label="L1:"+key_legend[0][1]+"%, L2:"+key_legend[0][2]+"%")
            else:
                ax[2, i].plot(x, f1_dict_drp[key], label="L1:"+key_legend[0][1]+"%, L2:"+key_legend[0][2]+"%")


ax[2, 3].legend(loc='center left', bbox_to_anchor = (1, 0.5), ncol=1, prop={'size': 10} )
ax[1, 3].legend(loc='center left', bbox_to_anchor = (1, 0.5), ncol=1, prop={'size': 10} )
ax[0, 3].legend(loc='center left', bbox_to_anchor = (1, 0.5), ncol=1, prop={'size': 10} )
fig.text(0.5, 0.04, "Iteration", ha='center')
fig.text(0.04, 0.5, 'F1 score', va='center', rotation='vertical')



plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
#plt.show()
#plt.savefig("figures/F1_variability_per_caller_dropout_NewBestModel.pdf", format = 'pdf', dpi=300)
plt.close()

for bm in best_models_f1_std:
    path = "Results_Hyperparameter_Tweaking_16042019/"+bm[3]+"/NA12878_CNN_results_"+"_".join(bm)
    f1_list =np.asarray(f1_dict["_".join(bm)])
    best_f1_index = np.argmax(f1_list)
    history = pickle.load(open(
        path + "/Training_Iteration_" + str(best_f1_index) + "/Best_Model_History_Iteration_" + str(best_f1_index),
        "rb"))
    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    x = range(0, len(acc))
    fig, ax1 = plt.subplots()
    ax1.set_title(
        "Best (high mean F1, low std) model of " + bm[3] + "\nSplit: " + bm[0] + ", epochs: " + bm[1] +
        ", learning rate: " + bm[2] + ", iteration: " + str(best_f1_index) + "\n F1:" + str(round(f1_list[best_f1_index], 3)))
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
    plt.savefig("figures/Best_highF1_lowSTD_model_of_" + bm[3], format='pdf', dpi=300)

    #plt.show()
    plt.close()

