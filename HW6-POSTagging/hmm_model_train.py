#hmm_model_train.py
#By Josie and Phuong
from collections import Counter
import sys
import pickle
from math import log, exp
import time
from decimal import *
# A = tag1, tag2 : prob
# B = word, tag : prob
# states = dict of states



def forward(A, B, states, obs):
    T = len(obs)
    N = len(states)
    alpha = [{} for i in range(T)]
    for s in states:
        firstWord = obs[0]

        aVal = A[('START', s)]
        bVal = B[(firstWord, s)]

        alpha[0][s] = aVal*bVal

    for t in range(1, T):
        for s in states:
            sum = 0
            all0 = True
            a0 = True
            b= True
            al0= True
            for sPrime in states:
                aVal = A[(sPrime, s)]
                bVal = B[(obs[t], s)]
                sum += alpha[t-1][sPrime]*aVal*bVal

            alpha[t][s] = sum


    sum = 0
    for s in states:
        aVal = A[(s, "END")]
        sum += alpha[T-1][s]*aVal

    return alpha, sum


def backward(A, B, states, obs):
    N = len(states)
    T = len(obs)
    beta = [{} for i in range(T)]
    for i in states:

        beta[T-1][i] = A[(i, "END")]
    for t in range(T-2, -1, -1):
        for i in states:
            sum = 0
            for j in states:
                aVal = A[(i,j)]
                bVal = B[(obs[t-1], j)]
                sum += beta[t+1][j]*aVal*bVal

            beta[t][i] = sum
    sum = 0
    for j in states:
        aVal = A[("START",j)]
        bVal = B[(obs[0], j)]

        sum += aVal*bVal*beta[0][j]


    return beta, sum



def notConverged(i):
    print(i)
    return (i<1000)


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
def forwardBackward(AllObs, V, Q, Amat, Bmat):
    A = Amat
    B = Bmat


    count = 0
    while notConverged(count):
        count += 100


        #init a hat and b hat to 0
        aHat = Counter()
        bHat = Counter()
        ObsCount = 0
        for Obs in AllObs[4:int(len(AllObs)/10)]:
            ObsCount += 1
            if ObsCount%50==0:
                print(ObsCount)
            alpha, alphaSum = forward(A, B, Q, Obs)
            beta, betaSum = backward(A, B, Q, Obs)
            print(alphaSum, betaSum)
            sys.exit()
            T = len(Obs)
            N = len(Q)

            gamma = [{} for i in range(T)]
            zeta = [{} for i in range(T)]

            #E-Step
            for t in range(T):
                for j in Q:

                    gamma[t][j] = (alpha[t][j]*beta[t][j])/alphaSum
                    for i in Q:
                        if t < T-1:
                            zeta[t][(i,j)] = (alpha[t][i]*A[(i,j)]*B[(Obs[t+1],j)]*beta[t+1][j])/alphaSum

            #M-Step
            for i in Q:
                for j in Q:
                    numerator = 0
                    denomenator = 0
                    for t in range(T-1):
                        numerator += zeta[t][(i,j)]
                        minidenom = 0
                        for k in Q:
                            minidenom += zeta[t][(i,k)]
                        denomenator += minidenom
                    aHat[(i,j)] += numerator/denomenator


            #bhat
            for j in Q:
                for vK in Obs:
                    numerator = 0
                    denomenator = 0
                    for t in range(T):
                        if Obs[t] == vK:
                            numerator += gamma[t][j]
                        denomenator += gamma[t][j]
                    bHat[(vK, j)] += numerator/denomenator


        #normalize A
        for i in Q:
            #get sum of state 1
            tot = 0
            for j in  Q:
                tot += aHat[(i,j)]
            for j in Q:
                A[(i,j)] = aHat[(i,j)]/tot

        print(B)
        #normalize B
        for i in Q:
            #get sum of state
            tot = 0
            for j in V:
                tot += bHat[(j, i)]
            for j in V:
                B[(j, i)] = bHat[(j, i)]/tot
                print(B[(j, i)])

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


def main():
    print("Training in process. Please wait...")
    with open('countmodel.dat', 'rb') as handle:
        matrixes = pickle.loads(handle.read())
    states = matrixes["allStates"]
    wds = matrixes["vocab"]
    if "<s>" in wds:
        del wds["<s>"]
    if "</s>" in wds:
        del wds["</s>"]
    vocab = list(wds.keys())

    dataSet = matrixes["allData"]

    vcb = {}
    for v in dataSet:
        for o in v:
            vcb[o] = 1
    vocab = list(vcb.keys())
    if "START" in states:
        del states["START"]
    if "END" in states:
        del states["END"]

    states = list(states.keys())
    bProb = 1/(len(vocab))
    aProb = 1/(len(states)+1)
    A = Counter({})
    B = Counter({})
    for state in states:
        A[("START", state)] = aProb

    for state1 in states:
        for state2 in states:
            A[(state1, state2)] = aProb
    for state in states:
        A[(state, "END")] = aProb

    #word given tag
    for word in vocab:
        for state in states:
            B[(word,state)] = bProb

    V = vocab
    Q = states


    aHat, bHat = forwardBackward(dataSet, V, Q, A, B)
    hmmCountModel = {}
        
    with open("trainmodel.dat", "wb") as outFile:

        hmmCountModel["aMatrix"] = aHat
        hmmCountModel["bMatrix"] = bHat
        pickle.dump(hmmCountModel, outFile)


        print("Training completed. Saving to model.dat")

main()