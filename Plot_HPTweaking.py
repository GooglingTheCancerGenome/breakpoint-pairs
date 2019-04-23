import matplotlib.pyplot as plt
import statistics
import re
import pickle
import numpy as np


callers = ["delly", "lumpy", "manta", "gridss"]
learning_rates = ["2", "3", "4", "5", "6"]
splits = ["20", "30", "40"]
epochs = ["10", "50", "100"]

f1_dict = {}
best_f1_per_caller = {}
best_std_per_caller = {}
for caller in callers:
    best_f1_per_caller[caller] = []
    best_std = 100
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
                f1_over10 = np.asarray(f1_over10)
                best_index = np.argmax(f1_over10)
                if std < best_std:
                    best_std = std
                    best_std_per_caller[caller] = ["_".join([split, epoch, lr, caller]), best_std, best_index, f1_over10[best_index]]
print(best_std_per_caller)
print(best_f1_per_caller)
x = range(0, 10)
fig, ax = plt.subplots(nrows=5, ncols=4, figsize=(12, 12), sharex=True, sharey=True)
fig.suptitle("F1 variability per caller, learning rate, epochs and split\nover 10 iterations")
for i, caller in enumerate(callers):
    ax[0, i].set_title(caller)
    for key in f1_dict:
        if "2_"+caller in key:
            if "50_2_"+caller in key:
                ax[0, i].plot(x, f1_dict[key], linestyle="dashdot")
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
            key_legend = re.findall(r'(\d*_\d*)_\d', key)
            if "50_4_"+caller in key:
                ax[2, i].plot(x, f1_dict[key], linestyle="dashdot", label=key_legend[0])
            elif "10_4_"+caller in key:
                ax[2, i].plot(x, f1_dict[key], linestyle="dashed", label=key_legend[0])
            else:
                ax[2, i].plot(x, f1_dict[key], label=key_legend[0])

    for key in f1_dict:
        if "5_"+caller in key:
            key_legend = re.findall(r'(\d*_\d*)_\d', key)
            if "50_5_"+caller in key:
                ax[3, i].plot(x, f1_dict[key], linestyle="dashdot", label=key_legend[0])
            elif "10_5_"+caller in key:
                ax[3, i].plot(x, f1_dict[key], linestyle="dashed", label=key_legend[0])
            else:
                ax[3, i].plot(x, f1_dict[key], label=key_legend[0])

    for key in f1_dict:
        if "6_"+caller in key:
            key_legend = re.findall(r'(\d*_\d*)_\d', key)
            if "50_6_"+caller in key:
                ax[4, i].plot(x, f1_dict[key], linestyle="dashdot", label=key_legend[0])
            elif "10_6_"+caller in key:
                ax[4, i].plot(x, f1_dict[key], linestyle="dashed", label=key_legend[0])
            else:
                ax[4, i].plot(x, f1_dict[key], label=key_legend[0])

ax[4, 3].legend(prop={'size': 5})
fig.text(0.5, 0.04, "Iteration", ha='center')
fig.text(0.04, 0.5, 'F1 score', va='center', rotation='vertical')
ax[0, 0].text(0.01, 0.04, 'Learning rate: 2', ha='left', va='top', rotation='horizontal')
ax[1, 0].text(0.01, 0.04, 'Learning rate: 3', ha='left', va='top', rotation='horizontal')
ax[2, 0].text(0.01, 0.04, 'Learning rate: 4', ha='left', va='top', rotation='horizontal')
ax[3, 0].text(0.01, 0.04, 'Learning rate: 5', ha='left', va='top', rotation='horizontal')
ax[4, 0].text(0.01, 0.04, 'Learning rate: 6', ha='left', va='top', rotation='horizontal')


plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
#plt.show()
plt.savefig("figures/F1_variability_per_caller.pdf", format = 'pdf', dpi=300)
plt.close()




for best in callers:
    history = pickle.load(open("Results_Hyperparameter_Tweaking_16042019/"+best+"/NA12878_CNN_results_"
                               +best_std_per_caller[best][0]+"/Training_Iteration_" + str(best_std_per_caller[caller][2])
                               + "/Best_Model_History_Iteration_" + str(best_std_per_caller[caller][2]), "rb"))
    best_hyperpar = re.findall(r'(\d*)_(\d*)_(\d)_(\S*)', best_std_per_caller[best][0])
    print(best_hyperpar)
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
    plt.plot(x, acc, color='red', label="Acc")
    plt.plot(x, val_acc, color='lightcoral', label="Val_Acc")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 10)
    plt.plot(x, loss, color='blue', label="Loss")
    plt.plot(x, val_loss, color='lightblue', label="Val_Loss")
    ax2.tick_params(axis='y')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    plt.savefig("figures/Best_low_std_model_of_"+best, format='pdf', dpi=300)

    #plt.show()
    plt.close()

