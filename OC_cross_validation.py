#!/usr/bin/env python

# Imports
import gzip
import os

import numpy as np

import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

import math

from collections import Counter, defaultdict

# Keras imports
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution1D, Flatten, MaxPooling1D
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K

from mcfly import modelgen, find_architecture

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize

# TensorFlow import
import tensorflow as tf

# Pandas import
import pandas as pd
import argparse
#import altair as alt

# Bokeh import
#from bokeh.io import show, output_file
#from bokeh.plotting import figure

import logging


def get_classes(labels):
    return sorted(list(set(labels)))


def get_channel_labels():
    # Fill labels for legend
    labels = []
    with open("Channel_Labels.txt","r") as inlab:
        for line in inlab:
            line = line.strip()
            labels += [line]
    return labels

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

def create_model(X, y_binary, lr):

    models = modelgen.generate_models(X.shape,
                                      y_binary.shape[1],
                                      number_of_models = 1,
                                      model_type = 'CNN',
                                      cnn_min_layers=2,
                                      cnn_max_layers=2,
                                      cnn_min_filters = 4,
                                      cnn_max_filters = 4,
                                      cnn_min_fc_nodes=6,
                                      cnn_max_fc_nodes=6,
                                      low_lr=lr, high_lr=lr,
                                      low_reg=1, high_reg=1,
                                      kernel_size=7)

    # models = modelgen.generate_models(X.shape,
    #                                   y_binary.shape[1],
    #                                   number_of_models = 1,
    #                                   model_type = 'CNN',
    #                                   cnn_min_layers=4,
    #                                   cnn_max_layers=4,
    #                                   cnn_min_filters = 6,
    #                                   cnn_max_filters = 6,
    #                                   cnn_min_fc_nodes=12,
    #                                   cnn_max_fc_nodes=12,
    #                                   low_lr=2, high_lr=2,
    #                                   low_reg=1, high_reg=1,
    #                                   kernel_size = 7)

    # i = 0
    # for model, params, model_types in models:
    #     logging.info('model ' + str(i))
    #     i = i + 1
    #     logging.info(params)
    #     model.summary()

    return models


def cross_validation(X, y, y_binary, channels, X_test, y_test, y_binary_test, output_dir_test, win_ids_test, split, epochs, lr):

    results = pd.DataFrame()
    X, y_binary = shuffle(X, y_binary, random_state=0)
    xtrain, xval, ytrain_binary, yval = train_test_split(X, y_binary,
                                                         test_size=split, random_state=2)
    #print(xtrain.shape)
    #print(xval.shape)
    #print(ytrain_binary.shape)
    #print(yval.shape)
    #print(ytrain_binary)
    #print(yval)
    for i in range(0, 10):
        logging.info("Training model " + str(i + 1) + "/10...")

        output_iter_dir = output_dir_test+'/Training_Iteration_' + str(i + 1)
        if not os.path.isdir(output_iter_dir):
            os.mkdir(output_iter_dir)

        # Clear model, and create it
        model = None
        model = create_model(X, y_binary, lr)

        # Debug message I guess
        logging.info ("Training new iteration on " + str(xtrain.shape[0]) + " training samples, " +
         str(xval.shape[0]) + " validation samples, this may take a while...")

        history, model = train_model(model, xtrain, ytrain_binary, xval, yval, epochs)

        model.save(output_iter_dir+"/Best_Model_Iteration_"+str(i+1)+".h5")
        with open(output_iter_dir+'/Best_Model_History_Iteration_'+str(i+1), 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        accuracy_history = history.history['acc']
        val_accuracy_history = history.history['val_acc']
        logging.info("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(
            val_accuracy_history[-1]))

        score_test = model.evaluate(X_test, y_binary_test, verbose=False)
        logging.info('Test loss and accuracy of best model: ' + str(score_test))

        results, probs = evaluate_model(model, X_test, y_test, y_binary_test, results, i, channels, output_iter_dir, epochs, hist=history.history,
                                 train_set_size=xtrain.shape[0],
                                 validation_set_size=xval.shape[0])

        with open(output_iter_dir+"/Called_Test_SVs.txt", "w") as out_sv:
            out_sv.write("Chromosome\tStart\tEnd\tProbs[DEL]\tProbs[No_DEL]\n")
            for k in range(0, len(win_ids_test)):
                out_sv.write("\t".join(["\t".join(win_ids_test[k].split("_")), str(probs[k][0]), str(probs[k][1])])+"\n")
    return results


def train_model(model, xtrain, ytrain, xval, yval, epochs):

    train_set_size = xtrain.shape[0]
    #print(xtrain.shape)
    #print(ytrain.shape)
    #print(xval.shape)
    #print(yval.shape)
    histories, val_accuracies, val_losses = find_architecture.train_models_on_samples(xtrain, ytrain,
                                                                                      xval, yval,
                                                                                      model, nr_epochs=epochs,
                                                                                      subset_size=train_set_size,
                                                                                      verbose=False)

    best_model_index = np.argmax(val_accuracies)
    best_model, best_params, best_model_types = model[best_model_index]
    #logging.info(best_model_index, best_model_types, best_params)

    nr_epochs = epochs
    history = best_model.fit(xtrain, ytrain,
                             epochs=nr_epochs, validation_data=(xval, yval),
                             verbose=False)

    return history, best_model


def evaluate_model(model, X_test, y_test, ytest_binary, results, cv_iter, channels, output_dir, epochs, hist,
                   train_set_size, validation_set_size):

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
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(ytest_binary[:, i],
                                                            probs[:, i])
        average_precision[i] = average_precision_score(ytest_binary[:, i], probs[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(ytest_binary.ravel(),
                                                                    probs.ravel())
    average_precision["micro"] = average_precision_score(ytest_binary, probs,
                                                         average="micro")
    average_precision["micro"] = average_precision_score(ytest_binary, probs,
                                                         average="micro")
    logging.info('Average precision score, micro-averaged over all classes: {0:0.2f}'
          .format(average_precision["micro"]))

    results = results.append({
        "channels": channels,
        "iter": cv_iter+1,
        "training_set_size": train_set_size,
        "validation_set_size": validation_set_size,
        "test_set_size": X_test.shape[0],
        "average_precision_score": average_precision["micro"],
        #"F1 Score":
    }, ignore_index=True)

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

    history = hist

    acc = history["acc"]
    loss = history["loss"]
    val_acc = history["val_acc"]
    val_loss = history["val_loss"]
    x = range(1, len(acc)+1)
    #print(x)
    #print(acc)
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='r')
    plt.plot(x, acc, color='red', label="Acc")
    plt.plot(x, val_acc, color='lightcoral', label="Val_Acc")
    ax1.tick_params(axis='y', color='r', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='blue')
    plt.plot(x, loss, color='blue', label="Loss")
    plt.plot(x, val_loss, color='lightblue', label="Val_Loss")
    ax2.tick_params(axis='y', color='blue', labelcolor='blue')
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    plt.savefig(output_dir+"/BestModelHistory_Plot.png", dpi=300, format='png')
    plt.close()

    # for iter_class in mapclasses.values():
    #
    #     predicted = probs.argmax(axis=1)
    #     #logging.info(predicted)
    #     y_pred_class = np.array([1 if i == iter_class else 0 for i in predicted])
    #     #logging.info(y_pred_class)
    #
    #     # keep probabilities for the positive outcome only
    #     probs_class = probs[:, iter_class]
    #     #logging.info(probs_class)
    #
    #     #logging.info(y_test)
    #
    #     y_test_class = np.array([1 if i[iter_class] == 1 else 0 for i in ytest_binary])
    #
    #     # calculate precision-recall curve
    #     precision, recall, thresholds = precision_recall_curve(y_test_class, probs_class)
    #     # calculate F1 score
    #     f1 = f1_score(y_test_class, y_pred_class)
    #     # calculate precision-recall AUC
    #     auc_value = auc(recall, precision)
    #     # calculate average precision score
    #     ap = average_precision_score(y_test_class, probs_class)
    #     logging.info('f1=%.3f auc=%.3f average_precision_score=%.3f' % (f1, auc_value , ap))
    #     # plot no skill
    #     plt.plot([0, 1], [0.5, 0.5], linestyle='--')
    #     # plot the roc curve for the model
    #     plt.plot(recall, precision, marker='.')
    #     # show the plot
    #     plt.savefig('Plots/Precision_Recall_multiclass_Iter_'+str(cv_iter)+'_'+channels+'.png', bbox_inches='tight')

    return results, probs


def run_cv(output_dir_test, datapath_training, datapath_test, split, epochs, lr):

    labels = get_channel_labels()

    results = pd.DataFrame()

    channels = 'all'
    logging.info('Running cv with '+channels+' channels:')
    for i, l in enumerate(labels):
        logging.info(str(i) + ':' + l)

    # Load the data
    X, y, y_binary, win_ids = data(datapath_training)

    X_test, y_test, y_binary_test, win_ids_test = data(datapath_test)

    results = results.append(cross_validation(X, y, y_binary, channels, X_test, y_test, y_binary_test, output_dir_test, win_ids_test, split, epochs, lr))

    logging.info(results)
    results.to_csv(output_dir_test+"/CV_results.csv", sep='\t')


def plot_results(output_dir_test):

    source = pd.read_csv(filepath_or_buffer=output_dir_test+'/CV_results.csv', delimiter='\t')

    import numpy as np
    import matplotlib.pyplot as plt

    means = source.groupby('channels')['average_precision_score'].agg(np.mean).sort_values()
    logging.info(means)
    std = source.groupby('channels')['average_precision_score'].agg(np.std)
    logging.info(std)
    ind = np.arange(len(list(means.index)))  # the x locations for the groups
    width = 0.50  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(ind, means, width, yerr=std)

    plt.ylabel('average_precision_score')
    plt.title('average_precision_score per channel set')
    plt.xticks(ind, list(means.index))
    plt.xticks(rotation=45,  horizontalalignment='right')
    #plt.yticks(np.arange(0.8, 1))
    #plt.legend((p1[0]), ('Bar'))
    plt.ylim(bottom=0.8)
    plt.tight_layout()

    plt.savefig(output_dir_test+'/Results.png', bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Training CNN on DELs')
    parser.add_argument('-spl', '--split', type=float,
                        default=0.4,
                        help="Fraction of validation set")
    parser.add_argument('-epo', '--epochs', type=int, default=10,
                        help="Number of epochs to be trained on")
    parser.add_argument('-lr', '--learningrate', type=int, default=2,
                        help="Exponent for learning rate parameter in McFly")
    parser.add_argument('-cal', '--caller', type=str, default='delly', help='Caller whose SVs are used for training.')


    args = parser.parse_args()

    HPC_MODE = False

    datapath_prefix = '/hpc/cog_bioinf/ridder/users/smehrem/breakpoint-pairs' if HPC_MODE else '/home/cog/smehrem/breakpoint-pairs/'

    if HPC_MODE:
        datapath_training = datapath_prefix + '/Processed/Test/' + \
                            date + '/TrainingData/'
        datapath_test = datapath_prefix + '/Processed/Test/' + \
                        date + '/TestData/'
    else:
        datapath_training = datapath_prefix + "N12878_DEL_TrainingData_"+args.caller+".npz"
        datapath_test = datapath_prefix + "N12878_DEL_TestData_"+args.caller+".npz"

    output_dir_test = 'NA12878_CNN_results_'+str(int(args.split*100))+'_'+str(args.epochs)+'_'+str(args.learningrate)+"_"+args.caller
    if not os.path.isdir(output_dir_test):
        os.mkdir(output_dir_test)


    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(
        format=FORMAT,
        filename=os.path.join(output_dir_test, 'logfile.log'),
        level=logging.INFO)

    run_cv(output_dir_test,  datapath_training, datapath_test, split=args.split, epochs=args.epochs, lr=args.learningrate)
    plot_results(output_dir_test)


if __name__ == '__main__':

    main()