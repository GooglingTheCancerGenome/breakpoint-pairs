with open("Parameters_HPC_LR5.txt", "w") as outfile:
    for split in [0.2, 0.3, 0.4]:
        for epoch in [10, 50, 100]:
            for lr in [5, 6]:
                for caller in ["delly", "manta","gridss", "lumpy"]:
                    outfile.write("\t".join([str(split), str(epoch), str(lr), caller])+"\n")