#hmm_model_build.py
#By Josie and Phuong
import random
from collections import Counter
import sys

class Token:
    def __init__(self, pos, word):
        self.pos = pos
        self.word = word
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
            if pos2 == 'END':
                self.wordWithTagCounts[('</s>', pos2)] += 1
                if pos2 in self.unigramPOSCounts:
                    self.unigramPOSCounts[pos2] += 1
                else:
                    self.unigramPOSCounts[pos2] = 1

    def calcTransitionProbs(self):
        self.transitionProbs = {}        
        for bigram in self.bigramPOSCounts:

          self.transitionProbs[bigram] = self.bigramPOSCounts[bigram]/self.unigramPOSCounts[bigram[0]]
        #print(self.transitionProbs[bigram])
        return self.transitionProbs
          
    def calcEmissionProbs(self):       
        self.emissionProbs = {}        
        for tuple in self.wordWithTagCounts:
            self.emissionProbs[tuple] = self.wordWithTagCounts[tuple]/self.unigramPOSCounts[tuple[1]]
        #print(self.transitionProbs[bigram])
        return self.emissionProbs
        



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

def tokenizeSet(set):
    tokens = []
    for sentence in set:
        tokens.append(tokenize(sentence))
    return tokens

def tokenize(sentence):
    tokens = [Token("START", "<s>")]
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
    trainingSet, testSet = extractSets(sys.argv[1])
    trainingTokens = tokenizeSet(trainingSet)
    model = CountHMM()
    model.getCounts(trainingTokens)
        
    with open("countmodel.dat", "w") as outFile:
        outFile.write("<A>")
        A  = model.calcTransitionProbs()
        B = model.calcEmissionProbs() 
        for bigram in A:
            outFile.write("{("+ bigram[0] + ", " + bigram[1]+ "):" + str(A[bigram]) + "}\n")
        outFile.write("<B>")
        for wordTag in B:
            outFile.write("(" + wordTag[0] + ", " + wordTag[1] + ") :" + str(B[wordTag])+ "}\n")

    print("Saving to model.dat")
main()