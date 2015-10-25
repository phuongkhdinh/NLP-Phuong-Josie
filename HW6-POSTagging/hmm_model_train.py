#hmm_model_train.py
#By Josie and Phuong
from collections import Counter
import sys
import pickle
from math import log, exp
import time
from decimal import *
import random

# A = tag1, tag2 : prob
# B = word, tag : prob
# states = dict of states
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
                    if obs != '<s>':
                        self.wordWithTagCounts[(obs, pos1)] += 1
                else:
                    self.unigramPOSCounts[pos1] = 1
                    if obs != '<s>':
                        self.wordWithTagCounts[(obs, pos1)] = 1
                if (pos1, pos2) in self.bigramPOSCounts:
                    self.bigramPOSCounts[(pos1, pos2)] += 1
                else:
                    self.bigramPOSCounts[(pos1, pos2)] = 1
            if pos2 == 'END':
                pass


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
        

def checkUnderflow(pieces, otherinfo):
    s = 1
    for piece in pieces:
        if piece == 0:
            print("piece is zero ", pieces)
            if otherinfo != None:
                print(otherinfo)
            sys.exit()
        s *= piece
    if s== 0:
        print("prod is zero ", pieces)
        if otherinfo != None:
            print(otherinfo)
        sys.exit()

def forward(A, B, Pi, states, obs):
    T = len(obs) - 1
    N = len(states)
    alpha = [{} for i in range(T+1)]
    for s in states:
        firstWord = obs[1]

        bVal = B[(firstWord, s)]
        alpha[1][s] = Pi[s]*bVal
    for t in range(1, T):
        for j in states:
            summa = 0
            bVal = B[(obs[t+1], j)]
            for i in states:
                aVal = A[(i,j)]
                summa += alpha[t][i]*aVal

            alpha[t+1][j] = summa*bVal

    sum = 0
    for j in states:
        sum += alpha[T][j]

    return alpha, sum


def backward(A, B, Pi, states, obs):
    N = len(states)
    T = len(obs) -1
    beta = [{} for i in range(T+1)]
    for i in states:
        beta[T][i] = 1
    for t in range(T-1, 0, -1):
        for i in states:
            sum = 0
            for j in states:
                aVal = A[(i,j)]
                bVal = B[(obs[t+1], j)]
                sum += beta[t+1][j]*aVal*bVal
            beta[t][i] = sum
    sum = 0
    for j in states:
        aVal = Pi[j]
        bVal = B[(obs[1], j)]
        sum += aVal*bVal*beta[1][j]


    return beta, sum



def notConverged(i):
    return (i<10)


# Steps:
# init A, B
# Iterate until convergence:
#     init ahat, bhat = A, B
#     for each obs in browncorpus:
#         get alpha, beta
#         get gamma, zeta
#         using gamma, zeta, a, b, change ahat, bhat
#     normalize ahat, bhat to be A, B


# O = Obs, V = Output vocab, Q = Hidden states (POS), A = transition, B = emission
def forwardBackward(AllObs, V, Q, Amat, Bmat, Pimat):


    A = Amat
    B = Bmat
    Pi = Pimat


    count = 0

    #insert empty obs for indexing purposes
    for Obs in AllObs:
        Obs.insert(0, "")
    while notConverged(count):
        count += 1
        if count % 10 == 0:
            print(A,B)

        #init a hat and b hat to 0
        aHat = Counter()
        bHat = Counter()
        piHat = Counter()
        ObsCount = 0
        for Obs in AllObs[:100]:
            ObsCount += 1

            #if ObsCount % 50 == 0:

            alpha, alphaSum = forward(A, B,Pi, Q, Obs)
            beta, betaSum = backward(A, B, Pi, Q, Obs)


            T = len(Obs)-1
            N = len(Q)


            gamma = [{} for i in range(T+1)]
            zeta = [{} for i in range(T+1)]


            #E-Step
            for t in range(1,T+1):
                for j in Q:
                    gamma[t][j] = (alpha[t][j]*beta[t][j])/alphaSum
                    for i in Q:
                        if t < T:
                            zeta[t][(i,j)] = (alpha[t][i]*A[(i,j)]*B[(Obs[t+1],j)]*beta[t+1][j])/alphaSum

            #M-Step
            for i in Q:
                for j in Q:
                    numerator = 0
                    denomenator = 0
                    for t in range(1,T):
                        numerator += zeta[t][(i,j)]
                        minidenom = 0
                        for k in Q:
                            minidenom += zeta[t][(i,k)]
                        denomenator += minidenom
                    aHat[(i,j)] += numerator/denomenator


            #bhat
            for j in Q:
                for vK in V:
                    numerator = 0
                    denomenator = 0
                    for t in range(1,T+1):
                        if Obs[t] == vK:
                            numerator += gamma[t][j]
                        denomenator += gamma[t][j]
                    bHat[(vK, j)] += numerator/denomenator

            for i in Q:
                piHat[i] += gamma[1][i]

        #normalize
        for i in Q:
            #get sum of state 1
            tot = 0
            for j in  Q:
                tot += aHat[(i,j)]
            for j in Q:
                if tot != 0:
                    A[(i,j)] = aHat[(i,j)]/tot


        #normalize B
        for i in Q:
            #get sum of state
            tot = 0
            for j in V:
                tot += bHat[(j, i)]
            for j in V:
                if tot != 0:
                 B[(j, i)] = bHat[(j, i)]/tot

        #normalize pi
        tot = 0
        for i in Q:
            tot += piHat[i]
        for i in Q:
            Pi[i] = piHat[i]/tot


    #put pi into a, and log probabilities
    for i in Q:
        A[("START", i)] = Pi[i]
    for i in Q:
        for j in Q:
            if A[(i,j)] == 0:
                A[(i,j)] = 0
            else:
                A[(i,j)] = log(A[(i,j)])
        for j in V:
            if B[(j,i)] == 0:
                B[(j,i)] = 0
            else:
                B[(j, i)] = log(B[(j, i)])
    return A, B



# Steps:
# init A, B
# Iterate until convergence:
#     init ahat, bhat = A, B
#     for each obs in browncorpus:
#         get alpha, beta
#         get gamma, zeta
#         using gamma, zeta, a, b, change ahat, bhat
#     normalize ahat, bhat to be A, B

def extractSets(filename):
    #Read data
    trainingSet = []
    initialSet = []
    with open(filename, "r") as f:
        lines = f.read().splitlines()
        for i in range(len(lines)-1):
            r = random.random()
            if r < 0.1:
                initialSet.append(lines[i])
            elif r>0.9:
                trainingSet.append(lines[i])

    return trainingSet, initialSet, lines

def tokenizeSet(sets):
    tokens = []
    for sentence in sets:
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
    print("Training in process. Please wait...")


    trainingSet, initialSet, allObs = extractSets("brown_tagged.dat")
    # Use initial set to generate guess of A and B
    initialTokens = tokenizeSet(initialSet) 
    model = CountHMM()
    model.getCounts(initialTokens)

    #Generate all vocabs
    V = set()
    vocabs = tokenizeSet(allObs)
    for vocab in vocabs:
        for v in vocab:
            V.add(v.word)
    #print(V)
    V.remove("</s>")
    V.remove("<s>")
    #Cal initial A,B,Q
    Atemp = Counter({})
    Pitemp = Counter({})
    completeTranmission = model.calcTransitionProbs()
    for transition in completeTranmission:
        if transition[0] == 'START':
            Pitemp[transition[1]] = completeTranmission[transition]
        else:
            Atemp[transition] = completeTranmission[transition]
    Btemp = model.calcEmissionProbs()
    tmp = model.unigramPOSCounts
    del tmp["START"]
    del tmp["END"]
    Q = list(tmp.keys())


    trainingSet = tokenizeSet(trainingSet) 
    trainingObs = []
    for obs in trainingSet:
        line = [x.word for x in obs if (x.word != "<s>" and x.word != "</s>")]
        trainingObs.append(line)


    A = Counter()
    for item in Atemp:
        #print(item)
        A[item] = exp(Atemp[item]) 

    B = Counter()
    for item in Btemp:
        B[item] = exp(Btemp[item]) 

    Pi = Counter()
    for item in Pitemp:
        Pi[item] = exp(Pitemp[item]) 


    N = len(Q)
    for i in Q:
        #get sum of state 1
        tot = 0
        zerot = 0
        for j in  Q:
            if A[(i,j)] == 0:
                zerot += 1
            tot += A[(i,j)]
        for j in Q:
            if tot != 0:
                if A[(i,j)] != 0:
                    if zerot != 0:
                        A[(i,j)] = A[(i,j)]*.9
                else:
                    A[(i,j)] = .1/zerot
            else:
                A[(i,j)] = 1/N


    for i in Q:
        #get sum of state 1
        tot = 0
        zerot = 0
        for j in  V:
            if B[j,i] == 0:
                zerot += 1
            tot += B[(j,i)]
        for j in V:
            if tot != 0:
                if B[(j,i)] != 0:
                    if zerot != 0:
                        B[(j,i)] = B[(j,i)]*.5
                else:
                    B[(j,i)] = .5/zerot
            else:
                B[(i,j)] = 1/N

    tot = 0
    zerot = 0
    for i in Q:
        if Pi[i] == 0:
            zerot += 1
        tot += Pi[i]
    for i in Q:
        if tot != 0:
            if Pi[i] != 0:
                if zerot != 0:
                    Pi[i] = .9*Pi[i]
            else:
                Pi[i] = .1/zerot
        else:
            Pi[i] = 1/N
    aHat, bHat = forwardBackward(trainingObs, V, Q, A, B, Pi)

    hmmCountModel = {}
        
    with open("trainmodel.dat", "wb") as outFile:

        hmmCountModel["aMatrix"] = aHat
        hmmCountModel["bMatrix"] = bHat
        pickle.dump(hmmCountModel, outFile)


        print("Training completed. Saving to trainmodel.dat")

main()