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
    T = len(obs) -1
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
    return (i<100)


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
        obsProbSum = 0
        for Obs in AllObs:
        #for Obs in AllObs[:10]:
            ObsCount += 1



            alpha, alphaSum = forward(A, B,Pi, Q, Obs)
            beta, betaSum = backward(A, B, Pi, Q, Obs)


            # if alphaSum != betaSum:
            #     print(count, alphaSum, betaSum)
                #sys.exit()


            T = len(Obs)-1
            N = len(Q)
            if alphaSum == 0:
                obsProbSum = float("inf")
            else:
                obsProbSum += 1/alphaSum
            # #eq 1
            # for i in Q:
            #     for j in Q:
            #         summa = 0
            #         for t in range(1,T):
            #             alpha[t][i]
            #             A[(i,j)]
            #             B[(Obs[t], i)]
            #             beta[t+1]
            #             summa+= alpha[t][i]*A[(i,j)]*B[(Obs[t+1], i)]*beta[t+1][i]
            #         aHat[(i,j)] += summa
            #
            # #eq 2
            # summa = 0
            # for i in Q:
            #     for sigma in Obs:
            #         summa = 0
            #         for t in range(1,T+1):
            #             if Obs[t] == sigma:
            #                 summa += alpha[t][i]*beta[t][i]
            #         bHat[(sigma, i)] += summa






            gamma = [{} for i in range(T+1)]
            zeta = [{} for i in range(T+1)]


            #E-Step
            for t in range(1,T+1):
                for j in Q:
                    #checkUnderflow([alpha[t][j],beta[t][j]], 50)
                    gamma[t][j] = (alpha[t][j]*beta[t][j])/alphaSum
                    for i in Q:
                        if t < T:
                            #checkUnderflow([alpha[t][i],A[(i,j)],B[(Obs[t+1],j)],beta[t+1][j]], 50)
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

        # print("________________")
        # #print(A)
        # Ast = A.copy()
        # #normalize A


        # #make up for eq1, 2
        # for i in Q:
        #     for j in Q:
        #         aHat[(i,j)] = aHat[(i,j)] * obsProbSum
        #     for sigma in V:
        #         bHat[(sigma, i)] = bHat[(sigma, i)] * obsProbSum


        # totstart = 0
        # totend = 0
        #

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
            zeroB = 0
            for j in V:
                tot += bHat[(j, i)]
            for j in V:
                if tot != 0:
                 B[(j, i)] = bHat[(j, i)]/tot
        tot = 0
        for i in Q:
            tot += piHat[i]
        for i in Q:
            Pi[i] = piHat[i]/tot




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
    states = matrixes["states"]
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
    A = Counter()
    B = Counter()
    Pi = Counter()
    for state in states:
        Pi[state] = aProb

    for state1 in states:
        for state2 in states:
            A[(state1, state2)] = aProb
    for state in states:
        A[(state, "END")] = aProb

    #word given tag
    for word in vocab:
        for state in states:
            B[(word,state)] = bProb
    # print(B)

    V = vocab
    Q = states


    V = ["N", "E"]
    Q = ["S1", "S2"]
    Pi = Counter()
    Pi["S1"] = 0.2
    Pi["S2"] = 0.8
    A = Counter()
    B = Counter()
    A[("S1", "S2")] = 0.5
    A[("S1", "S1")] = 0.5
    A[("S2", "S1")] = 0.3
    A[("S2", "S2")] = 0.7

    B[("N", "S1")] = 0.3
    B[("N", "S2")] = 0.8
    B[("E", "S1")] = 0.7
    B[("E", "S2")] = 0.2

    m = "NN,NN,NN,NN,NE,EE,EN,NN,NN"
    b = m.split(",")
    dataSet = []
    for i in b:
        sdd = []
        for c in i:
            if c == 'N' or c == 'E':
                sdd.append(c)
        dataSet.append(sdd)
    print(dataSet)
    aHat, bHat = forwardBackward(dataSet, V, Q, A, B, Pi)

    print(aHat)
    print(bHat)
    hmmCountModel = {}
        
    with open("trainmodel.dat", "wb") as outFile:

        hmmCountModel["aMatrix"] = aHat
        hmmCountModel["bMatrix"] = bHat
        pickle.dump(hmmCountModel, outFile)


        print("Training completed. Saving to model.dat")

main()