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
    if x>0:
        print("HOO")
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

        #TODO: FIX WITH UNK????
        #aVal = 0.0001 if ("START", s) not in A else A[("START", s)]
        #bVal = 0.0001 if (obs[0], s) not in B else B[(obs[0], s)] # if Run on corpus, need to handle Unknown here
        aVal = A[('START', s)]
        bVal = B[(obs[0], s)]
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
            bVal = B[(obs[t], s)]
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
                aVal = A[(i,j)]
                bVal = B[(obs[t+1], j)]
                if j in beta[t+1]:
                    sum = log_add([sum, beta[t+1][j]+aVal+bVal])


            beta[t][i] = sum
    sum = _NEG_INF
    for j in states:
        aVal = A[("START",j)]
        bVal = B[(obs[0], j)]
        if j in beta[0]:
            sum = log_add([sum, aVal+bVal+beta[0][j]])

    beta[0]["START"] = sum
    #print(sum)
    #WHY THE HECK IS IT POSITVE!!!!!!!!!
    return beta




def converged(i):
    return (1000-i == 0)

# O = Obs, V = Output vocab, Q = Hidden states (POS), A = transition, B = emission
def forwardBackward(Obs, V, Q, A, B):
    alpha = forward(A, B, Q, Obs)
    #print(alpha)
    beta = backward(A, B, Q, Obs)
    print(beta)
    T = len(Obs)
    N = len(Q)

    gamma = [{} for i in range(T)]
    zeta = [{} for i in range(T)]
    aHat = A
    bHat = B

    count = 100
    while not converged(count):
        count -= 1

        #E-Step
        for t in range(T-1):
            for j in Q:
                gamma[t][j] = (alpha[t][j]*beta[t][j])/alpha[-1]["END"]
                #print(gamma[t][j])
                for i in Q:
                    # print((alpha[t][i]))
                    # print(aHat[(i,j)])
                    # print(bHat[(Obs[t+1],j)])
                    # print(beta[t+1][j])
                    # print(alpha[T-1]["END"])
                    zeta[t][(i,j)] = (alpha[t][i]*aHat[(i,j)]*bHat[(Obs[t+1],j)]*beta[t+1][j])/alpha[T-1]["END"]
                    #print(zeta[t][(i,j)])
        #M-Step
        #ahat i j
        for i in Q:
            for j in Q:
                numerator = 0
                denomenator = 0
                for t in range(T-1):
                    numerator += zeta[t][(i,j)]
                    minidenom = 0
                    for k in Q:
                        #print(zeta[t][(i,k)])
                        minidenom += zeta[t][(i,k)]
                    denomenator += minidenom
                aHat[(i,j)] = numerator/denomenator

        #bhat
        for j in Q:
            for vK in Obs:
                numerator = 0
                denomenator = 0
                for t in range(T):
                    if Obs[t] == vK:
                        numerator += gamma[t][j]
                    print(denomenator, gamma[t][i])
                    denomenator += gamma[t][j]
                bHat[(vK, j)] = numerator/denomenator
    return aHat, bHat





def main():
    #modelFile = sys.argv[1]
    modelFile = "countmodel"
    # for line in sys.stdin:
    #     obs = line.split(" ")

    with open('countmodel.dat', 'rb') as handle:
        matrixes = pickle.loads(handle.read())
    A = Counter(matrixes["aMatrix"])
    B = Counter(matrixes["bMatrix"])

    stateGraph = []
    for state in A:
        if state[0] not in stateGraph:
            stateGraph.append(state[0])
        if state[1] not in stateGraph:
            stateGraph.append(state[1])
    stateGraph.remove("START")
    stateGraph.remove("END")
    obs = "The dog was orange .".lower().split(" ")
    V = set(stateGraph)
    Q = stateGraph
    ahat, bhat = forwardBackward(obs, V, Q, A, B)
    print(ahat)
    print("______")
    print(bhat)


main()