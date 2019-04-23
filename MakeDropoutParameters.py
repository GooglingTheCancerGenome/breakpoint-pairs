import itertools

best_models = [["0.3", "100", "5", "delly"], ["0.3", "50", "4", "lumpy"], ["0.4", "50", "4", "manta"], ["0.3", "50", "4", "gridss"]]

drpout_1 = ["0.0", "0.2", "0.5"]
drpout_2 = ["0.0", "0.2", "0.5"]

paras = itertools.product(drpout_1, drpout_2)

with open("Dropout_Test_Paras.txt", "w") as outfile:
    for par in paras:
        for bm in best_models:
            x = bm + [par[0], par[1]]
            outfile.write("\t".join(x)+"\n")
