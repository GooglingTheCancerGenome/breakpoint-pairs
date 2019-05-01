callers = ["delly", "gridss", "lumpy", "manta"]
hp_consensus = ["0.2", "100", "4", "0.4", "0.1"]
stacked = ["n", "y"]

with open("Parameters_StackedWindows.txt", "w") as outfile:
    for caller in callers:
        for stack in stacked:
            outfile.write(caller + "\t" + "\t".join(hp_consensus) + "\t" + stack + "\n")
