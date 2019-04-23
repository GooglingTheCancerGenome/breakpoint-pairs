import itertools
list1=['a','b','c']
list2=[1,2]

[zip(x,list2) for x in itertools.permutations(list1,len(list2))]

best_models = [["0.3", "100", "5", "delly"], ["0.3", "50", "4", "lumpy"], ["0.4", "50", "4", "manta"], ["0.3", "50", "4", "gridss"]]

drpout_1 = ["0.0", "0.2", "0.35", "0.5"]