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
    print(A)
    #print(B)
    T = len(obs)
    N = len(states)
    alpha = [[0 for i in range(T)] for j in range(N+1)]
    for s in range(N):
        firstWord = obs[0]

        aVal = A[('START', states[s])]
        bVal = B[(firstWord, states[s])]
        alpha[s][0] = aVal*bVal

    for t in range(1, T):
        for s in range(N):
            sum = 0
            bVal = B[(obs[t], states[s])]
            #if bVal == 0:
            #    print("ho")
            #    bVal == B[("<UNK>", states[s])]/1000000
            for sPrime in range(N):
                aVal = A[(states[sPrime], states[s])]                    
                sum += alpha[sPrime][t-1]*aVal*bVal
            
            alpha[s][t] = sum
            #print(alpha[s][T-7])

    #TEST WHERE THE SEQUENCE PROB OF WORDS BREAK!
    okay = [False] * T
    for t in range(1, T):
        for s in range(N):
            if alpha[s][t] > 0:
                okay[t]=True
    print(okay)

    #print(okay)
    sum = 0
    for s in range(N):
        aVal = A[(states[s], "END")]
        sum += alpha[s][T-1]*aVal


    return alpha, sum


def backward(A, B, states, obs):
    N = len(states)
    T = len(obs)
    beta = [[0 for i in range(T)] for j in range(N+1)] #beta[state][time]
    for i in range(N):
        beta[i][T-1] = A[(states[i], "END")]
    for t in range(T-2, -1, -1):
        for i in range(N):
            sum = 0
            for j in range(N):
                aVal = A[(states[i],states[j])]
                bVal = B[(obs[t-1], states[j])]
                sum += beta[j][t+1]*aVal*bVal

            beta[i][t] = sum
    sum = 0
    for j in range(N):
        aVal = A[("START",states[j])]
        bVal = B[(obs[0], states[j])]

        sum += aVal*bVal*beta[j][0]
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
        count += 500

        #init a hat and b hat to 0
        aHat = A
        bHat = B
        ObsCount = 0
        for Obs in AllObs[8:int(len(AllObs)/80)]:

            ObsCount += 1
            print(ObsCount)
            if ObsCount%50==0:
                print(ObsCount)
            alpha, alphaSum = forward(A, B, Q, Obs[:-1]) #Take out the end of line
            beta, betaSum = backward(A, B, Q, Obs[:-1])
            T = len(Obs) - 1
            N = len(Q)
            #gamma
            #gamma = [[0 for i in range(T)] for j in range(N+1)]
            # for state in range(N):
            #     for t in range(T):
            #         alpha[t][state] * beta[state][t][state] / alphaSum
            gamma = [[alpha[state][t] * beta[state][t] / alphaSum
                for t in range(T)]
            for state in range(N)] # array gamma[state][time]

            #zeta
            zeta = [[[alpha[state1][t]
                        * A[(Q[state1], Q[state2])]
                        * B[(Obs[t+1], Q[state2])]
                        * beta[state2][t+1]
                        / alphaSum
                  for t in range(T-1)]
              for state2 in range(N)]
            for state1 in range(N)] #zeta[si][sj][t]

            #M-step


            for i in range(N):
                tot = 0
                for j in range(N):
                    tot += sum(zeta[i][j])     # denom of aHat
                print(i, Q[i])    
                print(aHat)           
                aHat[("START", Q[i])] = sum(zeta[i][0]) / tot
                aHat[(Q[i], "END")] = sum(zeta[i][-1]) / tot
                for j in range(N):
                    aHat[(Q[i],Q[j])] = sum(zeta[i][j]) / tot #sum of all time / normalize
            #print(aHat)
            for i in range(N):
                for vocab in V:
                    num = sum(gamma[i][t] for t in range(T)
                                                  if Obs[t] == vocab)
                   # if n>0:
                       # print(n)
                    #if sum(gamma[i])>0:
                    bHat[(vocab, Q[i])] = num / sum(gamma[i]) # for all t = 1 to T

            A = aHat
            B = bHat


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
    #bProb = 0
    #aProb = 0

    A = Counter()
    for item in matrixes["aMatrix"]:
        A[item] = exp(matrixes["aMatrix"][item]) 

    B = Counter()
    for item in matrixes["bMatrix"]:
        B[item] = exp(matrixes["bMatrix"][item]) 
    #print(B)
    #sys.exit()
    #print(A,B)
    # A = Counter({})
    # B = Counter({})
    # for state in states:
    #     A[("START", state)] = aProb

    # for state1 in states:
    #    for state2 in states:
    #        A[(state1, state2)] = aProb
    # for state in states:
    #    A[(state, "END")] = aProb

    # # #word given tag
    # for word in vocab:
    #     for state in states:
    #         B[(word,state)] = bProb

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