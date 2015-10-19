#hmm_model_build.py
#By Josie and Phuong
import random


def extractSets(filename):
    #Read data
    trainingSet = []
    testSet = []
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        for i in range(len(lines)-1):
            r = random.random()
            if r < 0.1:
                testSet.append(lines[i])
            else:
                trainingSet.append(lines[i])

    return trainingSet, testSet

def main():
    trainingSet, testSet = extractSets("brown_tagged.dat")
    print(len(testSet))




main()