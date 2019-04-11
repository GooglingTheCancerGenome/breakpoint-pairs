import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

for splitvatriant in ["_60_40", "_80_20"]:

    results = pd.read_csv("NA12878_CNN_results"+splitvatriant+"/delly/CV_results.csv", delimiter="\t", index_col= 0)
    best_model_index = results["average_precision_score"].idxmax() + 1

    history = pickle.load(open("NA12878_CNN_results"+splitvatriant+"/delly/Training_Iteration_"+str(best_model_index)+"/Best_Model_History_Iteration_"+str(best_model_index), "rb"))

    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    x = range(0, 10)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    plt.plot(x, acc, color='red', label="Acc")
    plt.plot(x, val_acc, color='lightcoral', label="Val_Acc")
    ax1.tick_params(axis='y')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    plt.plot(x, loss, color='blue', label="Loss")
    plt.plot(x, val_loss, color='lightblue', label="Val_Loss")
    ax2.tick_params(axis='y')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1+h2, l1+l2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    plt.savefig("NA12878_CNN_results"+splitvatriant+"/delly/BestModelHistory_Plot.png", dpi=300, format='png')
    plt.close()