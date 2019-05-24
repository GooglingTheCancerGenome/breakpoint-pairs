#!/usr/bin/env python

from keras.models import load_model
import argparse
import pandas as pd
import logging
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
import os


def evaluate_model(model, X_test, y_test, ytest_binary, results,  output_dir, cv_iter):

    channels = "All"

    #Generate classes
    classes = sorted(list(set(y_test)))
    mapclasses = dict()
    for i, c in enumerate(classes):
        mapclasses[c] = i

    dict_sorted = sorted(mapclasses.items(), key=lambda x: x[1])
    #print(dict_sorted)
    # logging.info(dict_sorted)
    class_labels = [i[0] for i in dict_sorted]
    #print(class_labels)
    n_classes = ytest_binary.shape[1]
    # logging.info(ytest_binary)
    # logging.info(n_classes)
    probs = model.predict_proba(X_test, batch_size=1, verbose=False)

    # generate confusion matrix
    labels = sorted(list(set(y_test)))
    predicted = probs.argmax(axis=1)
    y_index = ytest_binary.argmax(axis=1)
    confusion_matrix = pd.crosstab(pd.Series(y_index), pd.Series(predicted))
    confusion_matrix.index = [labels[i] for i in confusion_matrix.index]
    confusion_matrix.columns = [labels[i] for i in confusion_matrix.columns]
    confusion_matrix.reindex(columns=[l for l in labels], fill_value=0)
    logging.info(confusion_matrix)
    confusion_matrix.to_csv(path_or_buf=output_dir+'/NA12878_confusion_matrix_cv_iter_' + str(cv_iter + 1) + '.csv')

    # print(np.diag(confusion_matrix))
    # print(confusion_matrix.sum(axis=1))
    print(confusion_matrix)
    # logging.info('Precision: %d' % int(np.diag(confusion_matrix) / confusion_matrix.sum(axis=1) * 100))
    # logging.info('Recall: %d' % int(np.diag(confusion_matrix)/confusion_matrix.sum(axis=0)*100))

    # For each class
    precision = dict()
    recall = dict()
    #f1 = dict()
    average_precision = dict()
    #average_f1 = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(ytest_binary[:, i],
                                                            probs[:, i])
        #f1[i] = (precision[i]*recall[i])/(precision[i]+recall[i])
        average_precision[i] = average_precision_score(ytest_binary[:, i], probs[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytest_binary.ravel(),
                                                                    probs.ravel())
    average_precision["micro"] = average_precision_score(ytest_binary, probs,
                                                         average="micro")

    #average_f1["micro"] = f1_score(ytest_binary, probs, average="micro")
    logging.info('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    #for key in f1:
        #logging.info("F1Score_"+str(key)+": "+str(f1[key]))

    results = results.append({
        "test_set_size": X_test.shape[0],
        "average_precision_score": average_precision["micro"]}, ignore_index=True)
    #for key in f1:
        #results = results.append({"f1_"+str(key): f1[key]}, ignore_index=True)

    plt.figure()
    plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall["micro"], precision["micro"], alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]))

    plt.savefig(output_dir+'/Precision_Recall_avg_prec_score_Iter_'+str(cv_iter)+'_'+channels+'.png', bbox_inches='tight')
    plt.close()

    from itertools import cycle
    # setup plot details
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (area = {0:0.2f})'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'
                      ''.format(class_labels[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.savefig(output_dir+'/Precision_Recall_avg_prec_score_per_class_Iter_' +
                str(cv_iter) +'_'+channels+'.png', bbox_inches='tight')
    plt.close()

    return results, probs

def data(datapath):


    dataset_type = '_balanced'
    data_input_file = datapath

    npzfiles = np.load(data_input_file)
    X = npzfiles['X']
    y = npzfiles['y']
    y_binary = npzfiles['y_binary']
    win_ids = npzfiles['ids']

    # logging.info(X.shape)
    # logging.info(y.shape)
    # logging.info(y.shape)
    # logging.info(win_ids.shape)

    # idx = np.arange(0,9)
    # idx = np.append(idx, np.arange(33,35))
    # idx = np.append(idx, np.arange(41, 44))
    # idx = np.append(idx,[12,16,20,24,28,32])

    return X, y, y_binary, win_ids


def run_cv(output_dir_test, datapath_test, model, cv_iter):

    results = pd.DataFrame()
    # Load the data

    X_test, y_test, y_binary_test, win_ids_test = data(datapath_test)

    results, probs = evaluate_model(model, X_test, y_test, y_binary_test, results, output_dir_test, cv_iter)

    with open(output_dir_test + "/Called_Test_SVs.txt", "w") as out_sv:
        out_sv.write("Chromosome\tStart\tEnd\tProbs[DEL]\tProbs[No_DEL]\n")
        for k in range(0, len(win_ids_test)):
            out_sv.write("\t".join(["\t".join(win_ids_test[k].split("_")), str(probs[k][0]), str(probs[k][1])]) + "\n")

    logging.info(results)
    results.to_csv(output_dir_test+"/CV_results.csv", sep='\t')

def main():

    parser = argparse.ArgumentParser(description='Training CNN on DELs')
    parser.add_argument('-inmod', '--input_model', type=str, default="",
                        help="Trained model to be tested on (path)")
    parser.add_argument('-cal', '--caller', type=str, default="",
                        help='Caller for test and model.')
    parser.add_argument('-neg', '--negativeset', type=str, default="",
                        help='What negative set is used.')
    parser.add_argument('-iter', '--iteration', type=int, default=1,
                        help='Iteration of Model')


    args = parser.parse_args()
    channels = []
    with open("Channel_Labels.txt", "r") as inchannel:
        for line in inchannel:
            line = line.strip()
            channels += [line]

    model = load_model(args.input_model)

    HPC_MODE = True

    datapath_prefix = '/hpc/cog_bioinf/ridder/users/smehrem/breakpoint-pairs' if HPC_MODE else '/home/cog/smehrem/breakpoint-pairs/'

    for channel in channels:
        datapath_test = datapath_prefix + "Test_Training_Data/shuffled/"+args.negativeset+"/N12878_DEL_TestData_"+channel+"_"+args.caller+".npz"
        output_dir_test = datapath_prefix + "/Channelimportance_Results_23052019/" + args.negativeset + "/" + 'NA12878_CNN_results_Consensus_' + args.caller + "_model" + "_"+str(args.iteration)+"_"+channel
        if not os.path.isdir(output_dir_test):
            os.makedir(output_dir_test)
        run_cv(output_dir_test, datapath_test, model, args.iteration)


if __name__ == '__main__':

    main()
