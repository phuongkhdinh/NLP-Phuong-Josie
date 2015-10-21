#hmm_model_train.py
#By Josie and Phuong
from collections import Counter

# A = tag1, tag2 : prob
# B = word, tag : prob
# states = dict of states

def forward(A, B, states, obs):
    T = len(obs)
    N = len(states)
    alpha = [{}]*J
    for s in states:

        #TODO: FIX WITH UNK????
        aVal = 0 if ("START", s) not in A else A[("START", s)]
        bVal = 0 if (obs[0], s) not in B else B[(obs[0], s)]
        alpha[0][s] = aVal*bVal
    for t in range(1, N):
        for s in states:
            sum = 0
            for sPrime in states:
                aVal = 0 if (sPrime,s) not in A else A[(sPrime,s)]
                bVal = 0 if (obs[t], s) not in B else B[(obs[t], s)]
                sum += alpha[(t-1, sPrime)]*aVal*bVal

            alpha[t][s] = sum
    sum = 0
    for s in states:
        aVal = 0 if (s, "END") not in A else A[(s, "END")]
        sum += alpha[N-1][s]*aVal
    alpha[N-1]["END"] = sum
    return alpha


def backward(A, states, obs):
    N = len(states)
    T = len(obs)
    beta = [{}]*T
    for i in states:
        beta[T-1][i] = A[(i, "END")]
    for t in range(1, T):
        for i in states:
            sum = 0
            for j in states:
                aVal = 0 if (i,j) not in A else A[(i,j)]
                bVal = 0 if (obs[t+1], j) not in B else B[(obs[t+1], j)]
                sum += beta[(t+1, j)]*aVal*bVal

            beta[t][i] = sum
    sum = 0
    for j in states:
        aVal = 0 if ("START",j) not in A else A[("START",j)]
        bVal = 0 if ("<s>", j) not in B else B[("<s>", j)]
        sum += beta[(t+1, j)]*aVal*bVal*beta[0][j]
    beta[0]["START"] = sum
    return beta




def converged(i):
    return (100-i == 0)

# O = Obs, V = Output vocab, Q = Hidden states (POS), A = transition, B = emission
def forwardBackward(Obs, V, Q, A, B):
    alpha = forward(A, B, Q, Obs)
    beta = backward(A, Q, Obs)
    T = len(Obs)
    N = len(Q)

    gamma = [{}]*T
    zeta = [{}]*T
    aHat = {}
    bHat = {}

    i = 100
    while not converged(i):
        i -= 1

        #E-Step
        for t in range(T):
            for j in Q:
                gamma[t][j] = (alpha[t][j]*beta[t][j])/alpha[-1]["END"]
                for i in Q:
                    zeta[(i,j)] = (alpha[t][i]*A[i][j]*B[(obs[t+1],j)]*beta[t+1][j])/alpha[T-1]["END"]
        #M-Step
        #ahat i j
        for i in Q:
            for j in Q:
                numerator = 0
                denomenator = 0
                for t in range(T-1):
                    numerator += zeta[t][(i,j)]
                    minidenom = 0
                    for k in states:
                        minidenom += zeta[t][(i,k)]
                    denomenator += minidenom
                ahat[(i,j)] = numerator/denomenator

        #bhat
        for j in Q:
            for vK in Obs:
                numerator = 0
                denomenator = 0
                for t in range(T):
                    if obs[t] == vK:
                        numerator += gamma[t][j]
                    denomenator += gamma[t][j]
                bhat[(vK, j)] = numerator/denomenator
    return ahat, bhat










def main():



main()