#hmm_model_build.py
#By Josie and Phuong
import random

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
        self.bicounts = {}
        self.counts = {}
        self.totalBigrams = 0
        self.totalUnigrams = 0
        token1 = ""
        token2 = ""
        for token in tokens:
            self.totalBigrams += (len(token) - 1)
            self.totalUnigrams += len(token)
            for i in range(len(token) - 1):
                token1 = token[i].pos
                token2 = token[i+1].pos
                if token1 in self.counts:
                    self.counts[token1] += 1
                else:
                    self.counts[token1] = 1
                if (token1, token2) in self.bicounts:
                    self.bicounts[(token1, token2)] += 1
                else:
                    self.bicounts[(token1, token2)] = 1
            if token2 == 'END':
                if token2 in self.counts:
                    self.counts[token2] += 1
                else:
                    self.counts[token2] = 1

    def getTransitionProbs(self):

        #maybe need to change range??
        self.transitionProbs = [[0 for i in range(len(self.counts))] for j in range(len(self.counts))]
        for bigram in self.bicounts:


        for bigram in self.bicounts:
            pass


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
        tokens.append(self.tokenize(sentence))
    return tokens

def tokenize(sentence):
    tokens = [Token("START", "<s>")]
    words = (sentence.split(" "))
    for word in word:
        tkn = word.split("/")
        token = Token(tkn[1], tkn[0])
        tokens.append(token)
    tokens.append(Token("END", "</s>"))
    return tokens



def main():


    trainingSet, testSet = extractSets("brown_tagged.dat")
    trainingTokens = tokenizeSet(trainingSet)
    model.getCounts(trainingTokens)




main()