#hmm_model_build.py
#By Josie and Phuong
import random
from collections import Counter
import sys
import pickle
from math import log

class Token:
    def __init__(self, pos, word):
        self.pos = pos
        self.word = word.lower()
class CountHMM:
    def __init__(self):
        return


     ##
        # Takes in a list of tokenized sentences and counts the total number of unigrams
        # and bigrams. Also counts the frequency of each unigram and bigram
        #
        # Taken from previous work with Ibrahim
    # #
    def getCounts(self, tokens):
        self.bigramPOSCounts = Counter()
        self.unigramPOSCounts = Counter()
        self.wordWithTagCounts = Counter()
        self.words = Counter()
        self.words["<UNK>"] = 0
        self.totalBigrams = 0
        self.totalUnigrams = 0
        pos1 = ""
        pos2 = ""
        obs = ""
        for token in tokens:
            self.totalBigrams += (len(token) - 1)
            self.totalUnigrams += len(token)
            for i in range(len(token) - 1):
                pos1 = token[i].pos
                pos2 = token[i+1].pos
                obs = token[i].word
                #if obs == "START":
                    #print("SHRW")
                if obs not in self.words:
                    self.words["<UNK>"] += 1
                    self.words[obs] = 0
                    obs = "<UNK>"

                if pos1 in self.unigramPOSCounts:
                    self.unigramPOSCounts[pos1] += 1
                    self.wordWithTagCounts[(obs, pos1)] += 1
                else:
                    self.unigramPOSCounts[pos1] = 1
                    self.wordWithTagCounts[(obs, pos1)] = 1
                if (pos1, pos2) in self.bigramPOSCounts:
                    self.bigramPOSCounts[(pos1, pos2)] += 1
                else:
                    self.bigramPOSCounts[(pos1, pos2)] = 1

            # #do last token
            # obs = token[-1].word
            # if obs not in self.words:
            #     self.words["<UNK>"] += 1
            #     self.words[obs] = 0
            #     obs = "<UNK>"
            # if pos2 in self.unigramPOSCounts:
            #     self.unigramPOSCounts[pos2] += 1
            #     self.wordWithTagCounts[(obs, pos2)] += 1
            # else:
            #     self.unigramPOSCounts[pos2] = 1
            #     self.wordWithTagCounts[(obs, pos2)] = 1

            if pos2 == 'END':
                #print("huh")
                self.wordWithTagCounts[('</s>', pos2)] += 1
                if pos2 in self.unigramPOSCounts:
                    self.unigramPOSCounts[pos2] += 1
                else:
                    self.unigramPOSCounts[pos2] = 1


    def calcTransitionProbs(self):
        self.transitionProbs = Counter({})        
        for bigram in self.bigramPOSCounts:

          self.transitionProbs[bigram] = log(self.bigramPOSCounts[bigram]) - log(self.unigramPOSCounts[bigram[0]])
        #print(self.transitionProbs[bigram])
        return self.transitionProbs
          
    def calcEmissionProbs(self):       
        self.emissionProbs = Counter({})        
        for tuple in self.wordWithTagCounts:
            self.emissionProbs[tuple] = log(self.wordWithTagCounts[tuple]) - log(self.unigramPOSCounts[tuple[1]])
        #print(self.transitionProbs[bigram])
        return self.emissionProbs
        



def extractSets(filename):
    #Read data
    trainingSet = []
    testSet = []
    lines = []
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        for i in range(len(lines)-1):
            r = random.random()
            if r < 0.1:
                testSet.append(lines[i])
            else:
                trainingSet.append(lines[i])

    return trainingSet, testSet, lines

def tokenizeSet(set):
    tokens = []
    for sentence in set:
        tokens.append(tokenize(sentence))
    return tokens

def tokenize(sentence):
    tokens = [Token("START", "<s>")]
    #tokens = []
    words = (sentence.split(" "))
    for word in words[:-1]:
        tkn = word.rsplit("/", 1)
        token = Token(tkn[1], tkn[0])
        tokens.append(token)
    tokens.append(Token("END", "</s>"))
    return tokens



def main():
    print("Calculating HMM probabilities....")
    
    #TODO: COMMANDLINE ARG
    trainingSet, testSet, all = extractSets("brown_tagged.dat") # extractSets(sys.argv[1])#
    trainingTokens = tokenizeSet(trainingSet)
    allTokens = tokenizeSet(all)
    allObs = []
    for obs in allTokens:
        line = [x.word for x in obs if (x.word != "<s>" and x.word != "/<s>")]
        allObs.append(line)
    model = CountHMM()
    model.getCounts(trainingTokens)

    countModel = {}

    with open("countmodel.dat", "wb") as outFile:
        #outFile.write("<A>")

        countModel["aMatrix"] = model.calcTransitionProbs()
        countModel["bMatrix"] = model.calcEmissionProbs()
        countModel["states"] = model.unigramPOSCounts
        countModel["vocab"] = model.words
        countModel["vocab"].remove("<UNK>")
        countModel["allData"] = allObs
        #print(countModel["bMatrix"])
        pickle.dump(countModel, outFile)
        # for bigram in A:
        #     outFile.write("{("+ bigram[0] + ", " + bigram[1]+ "):" + str(A[bigram]) + "}\n")
        # outFile.write("<B>")
        # for wordTag in B:
        #     outFile.write("{(" + wordTag[0] + ", " + wordTag[1] + ") :" + str(B[wordTag])+ "}\n")


        print("Saving to model.dat")
main()