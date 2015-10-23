#hmm_model_train.py
#By Josie and Phuong
from collections import Counter
import sys
import pickle
from math import log, exp

# A = tag1, tag2 : prob
# B = word, tag : prob
# states = dict of states

_NEG_INF = float('-inf')
def log_add(values):
    """
    Unlog values, Adds the values, returning the log of the addition.
    """
    x = max(values)
    if x > _NEG_INF:
        sumDiffs = 0
        for value in values:
            sumDiffs += exp(value - x)
        return x + log(sumDiffs)
    else:
        return x


def forward(A, B, states, obs):
    T = len(obs)
    N = len(states)
    alpha = [{} for i in range(T)]

    for s in states:
        firstWord = obs[0]
        #if (firstWord,s) not in B: #We have the first word with this tag
        #    firstWord = "<UNK>"
        #TODO: FIX WITH UNK????
        #aVal = 0.0001 if ("START", s) not in A else A[("START", s)]
        #bVal = 0.0001 if (obs[0], s) not in B else B[(obs[0], s)] # if Run on corpus, need to handle Unknown here
        aVal = A[('START', s)]
        bVal = B[(firstWord, s)]
        if aVal!= 0 and bVal != 0:
            alpha[0][s] = aVal + bVal

    for t in range(1, T):
        for s in states:
            sum = _NEG_INF
            for sPrime in states:
                #aVal = 0.0001 if (sPrime,s) not in A else A[(sPrime,s)]
                #bVal = 0.0001 if (obs[t], s) not in B else B[(obs[t], s)]
                if sPrime in alpha[t-1]:
                    aVal = A[(sPrime,s)]
                    sum = log_add([sum, alpha[t-1][sPrime]+aVal])
                #print(alpha[t-1][sPrime], aVal, bVal)
                # if bVal == 0:
                #     print(obs[t], s)
            word = obs[t]
            #if (word, s) not in B:
            #    word = "<UNK>"
            bVal = B[(word, s)]
            alpha[t][s] = sum + bVal

    sum = _NEG_INF
    for s in states:
        if s in alpha[T-1]:
            aVal = A[(s, "END")]
            sum = log_add([sum, alpha[T-1][s]+aVal])
    alpha[T-1]["END"] = sum
    # if alpha[T-1]["END"] == 0:
    #     print(alpha[T-1])
    return alpha


def backward(A, B, states, obs):
    N = len(states)
    T = len(obs)
    beta = [{} for i in range(T)]
    for i in states:
        if A[(i, "END")] != 0:
            beta[T-1][i] = A[(i, "END")]
    for t in range(T-2, -1, -1):
        for i in states:
            sum = _NEG_INF
            for j in states:
                if j in beta[t+1]:
                    aVal = A[(i,j)]
                    wd = obs[t+1]
                    #if (wd, j) not in B:
                    #    wd = "<UNK>"
                    bVal = B[(wd, j)]
                    sum = log_add([sum, beta[t+1][j]+aVal+bVal])


            beta[t][i] = sum
    sum = _NEG_INF
    for j in states:
        if j in beta[0]:
            aVal = A[("START",j)]
            wd = obs[0]
            #if (obs[0], j) not in B:
            #    wd = "<UNK>"
            bVal = B[(wd, j)]
            sum = log_add([sum, aVal+bVal+beta[0][j]])

    beta[0]["START"] = sum
    return beta




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


    print(AllObs)


    count = 0
    while notConverged(count):
        count += 1000

        aHat = A
        bHat = B
        ObsCount = 0
        for Obs in AllObs:
            ObsCount += 1
            if ObsCount%50==0:
                print(ObsCount)
            alpha = forward(A, B, Q, Obs)
            beta = backward(A, B, Q, Obs)
            T = len(Obs)
            N = len(Q)

            gamma = [{} for i in range(T)]
            zeta = [{} for i in range(T)]

            #E-Step
            for t in range(T-1):
                for j in Q:
                    if j in alpha[t] and j in beta[t]:
                        #print(beta[t][j])
                        gamma[t][j] = alpha[t][j] + beta[t][j] - alpha[-1]["END"]
                        #print(gamma[t][j])
                        for i in Q:
                            if i in alpha[t] and j in beta[t+1]:
                                # print((alpha[t][i]))
                                # print(aHat[(i,j)])
                                # print(bHat[(Obs[t+1],j)])
                                # print(beta[t+1][j])
                                # print(alpha[T-1]["END"])
                                wd = Obs[t+1]

                                #if (wd, j) not in bHat:
                                #    wd = "<UNK>"
                                zeta[t][(i,j)] = alpha[t][i] + aHat[(i,j)] + bHat[(wd,j)] + beta[t+1][j] - alpha[T-1]["END"]
                            #print(zeta[t][(i,j)])
            #M-Step
            #ahat i j
            for i in Q:
                for j in Q:
                    numerator = _NEG_INF
                    denomenator = _NEG_INF
                    for t in range(T-1):
                        if (i,j) in zeta[t]:
                            numerator = log_add([numerator, zeta[t][(i,j)]])
                            minidenom = _NEG_INF
                            for k in Q:
                                #print(zeta[t][(i,k)])
                                if (i,k) in zeta[t]:
                                    minidenom = log_add([minidenom, zeta[t][(i,k)]])
                            denomenator = log_add([denomenator, minidenom])
                    aHat[(i,j)] = numerator-denomenator

            #bhat
            for j in Q:
                for vK in Obs:
                    numerator = _NEG_INF
                    denomenator = _NEG_INF
                    for t in range(T):
                        if j in gamma[t]:
                            if Obs[t] == vK:
                                numerator = log_add([numerator, gamma[t][j]])
                            #print(denomenator, gamma[t][i])
                            denomenator = log_add([denomenator, gamma[t][j]])
                    bHat[(vK, j)] = numerator - denomenator

        #normalize A
        for i in Q:
            #get sum of state 1
            tot = []
            for j in  Q:
                tot.append(aHat[(i,j)])
            tot = log_add(tot)
            for j in Q:
                A[(i,j)] = aHat[(i,j)] - tot

        #normalize B
        for i in Q:
            #get sum of state
            tot = []
            for j in V:
                tot.append(bHat[(j, i)])
            tot = log_add(tot)
            for j in V:
                B[(j, i)] = bHat[(j, i)]- tot




    return A, B



#TODO: UNK
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
    states = matrixes["states"]
    vocab = list(matrixes["vocab"].keys())
    dataSet = matrixes["allData"]
    if "START" in states:
        del states["START"]
    if "END" in states:
        del states["END"]
    states = list(states.keys())
    print(states)
    bProb = log(1/(len(vocab)))
    aProb = log(1/(len(states)+1))
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
    print(bHat)
    hmmCountModel = {}
        
    with open("trainmodel.dat", "wb") as outFile:
        #outFile.write("<A>")

        hmmCountModel["aMatrix"] = aHat
        hmmCountModel["bMatrix"] = bHat
        pickle.dump(hmmCountModel, outFile)


        print("Training completed. Saving to model.dat")

main()