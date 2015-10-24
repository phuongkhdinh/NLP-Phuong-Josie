from collections import Counter
import sys
import pickle
from math import log, exp
import time
from decimal import *

def forward(obs, stateGraph, aMatrixDict, bMatrixDict):
	aMatrix = Counter(aMatrixDict)
	bMatrix = Counter(bMatrixDict)
	N = len(stateGraph)
	T = len(obs)
	#viterbi = {}
	#backpointer = {}
	alpha = [[0 for col in range(T)] for row in range(N+1)]	 #Matrix [N+2, T]
	#aMatrix in form {(qi, qj): double prob} given tag i, prob of tagj
	#bMatrix in form {(w, t): double prob} given tag t, prob of w
	#initialization
	for state in range(0, N): #MIGHT be N-1 or 1->N

		#if bMatrix[obs[0], stateGraph[state]] == 0: #We have the first word with this tag
		#	firstWord = "<UNK>"
		#try:
			#viterbi[(stateGraph[state], 0)] = math.log(aMatrix[("START",stateGraph[state])]) + math.log(bMatrix[(obs[0], stateGraph[state])])
		#	viterbi[state][0] = aMatrix[("START",stateGraph[state])] + bMatrix[(firstWord, stateGraph[state])] - 10
		#except:
		#else:

			alpha[state][0] = aMatrix[("START",stateGraph[state])] * bMatrix[(obs[0], stateGraph[state])]
		#elif ('START',stateGraph[state]) in aMatrix:
		# 	viterbi[state][0] = aMatrix[("START",stateGraph[state])] - 99

		#backpointer[state][0] = [stateGraph[state]]
	#Recursion
	for time in range(1,T): #MIGHT BE T-1
		for state in range(0,N):
			summation = 0
			for statePrime in range(0,N):
				#if (obs[time],stateGraph[state]) in bMatrix: #and (stateGraph[statePrime], stateGraph[state]) in aMatrix:
				#if alpha[statePrime][time-1] != None:
				#if bMatrix[obs[time], stateGraph[state]] == 0: #We have the first word with this tag
				#	word = "<UNK>"
				#	calVi = viterbi[statePrime][time-1] + \
				#	aMatrix[(stateGraph[statePrime], stateGraph[state])] + bMatrix[(word,stateGraph[state])] - 10
				#else:
					cal = alpha[statePrime][time-1] * \
					aMatrix[(stateGraph[statePrime], stateGraph[state])] * bMatrix[(obs[time],stateGraph[state])]
					summation += cal
			alpha[state][time] = summation
	# Termination
	#viterbi[N][T] = 0
	summationEnd = 0
	for state in range(0,N):
		if (stateGraph[state], "END") in aMatrix:
			calEnd = alpha[state][T-1] + aMatrix[(stateGraph[state], "END")]
			summationEnd += calEnd
		alpha[N][T-1] = summationEnd
	return alpha


def backward(obs, stateGraph, aMatrixDict, bMatrixDict):
	aMatrix = Counter(aMatrixDict)
	bMatrix = Counter(bMatrixDict)
	N = len(stateGraph)
	T = len(obs)
	#viterbi = {}
	#backpointer = {}
	beta = [[0 for col in range(T)] for row in range(N+1)]	 #Matrix [N+2, T]
	#aMatrix in form {(qi, qj): double prob} given tag i, prob of tagj
	#bMatrix in form {(w, t): double prob} given tag t, prob of w
	#initialization
	for i in range(0, N): #MIGHT be N-1 or 1->N

		#if bMatrix[obs[0], stateGraph[state]] == 0: #We have the first word with this tag
		#	firstWord = "<UNK>"
		#try:
			#viterbi[(stateGraph[state], 0)] = math.log(aMatrix[("START",stateGraph[state])]) + math.log(bMatrix[(obs[0], stateGraph[state])])
		#	viterbi[state][0] = aMatrix[("START",stateGraph[state])] + bMatrix[(firstWord, stateGraph[state])] - 10
		#except:
		#else:
			beta[i][T-1] = aMatrix[(stateGraph[i],"END")] 
			#alpha[state][0] = aMatrix[("START",stateGraph[state])] * bMatrix[(obs[0], stateGraph[state])]
		#elif ('START',stateGraph[state]) in aMatrix:
		# 	viterbi[state][0] = aMatrix[("START",stateGraph[state])] - 99

		#backpointer[state][0] = [stateGraph[state]]
	#Recursion
	for time in range(T-1): #MIGHT BE T-1
		for i in range(0,N):
			summation = 0
			for j in range(0,N):
				#if (obs[time],stateGraph[state]) in bMatrix: #and (stateGraph[statePrime], stateGraph[state]) in aMatrix:
				#if alpha[statePrime][time-1] != None:
				#if bMatrix[obs[time], stateGraph[state]] == 0: #We have the first word with this tag
				#	word = "<UNK>"
				#	calVi = viterbi[statePrime][time-1] + \
				#	aMatrix[(stateGraph[statePrime], stateGraph[state])] + bMatrix[(word,stateGraph[state])] - 10
				#else:
					summation += aMatrix[(stateGraph[i], stateGraph[j])] \
					                      * bMatrix[(obs[time+1],stateGraph[j])] * beta[j][time+1]
			beta[i][time] = summation
	# Termination
	#viterbi[N][T] = 0
	summationEnd = 0
	for state in range(0,N):
		if ("START", stateGraph[state]) in bMatrix:
			summationEnd += aMatrix[("START", stateGraph[state])] * bMatrix[(obs[0], stateGraph[state])] \
							* beta[state][0]
		beta[0][0] = summationEnd
	return beta


def main():
    print("Training in process. Please wait...")
    #getcontext().prec = 500
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
    #print(states)
    bProb = 1/(len(vocab))
    #print(bProb)
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

    aHat = forward(dataSet[270], Q, A, B)
    print(aHat)
    bHat = backward(dataSet[270], Q, A, B)
    print(bHat)
    #aHat, bHat = forwardBackward(dataSet, V, Q, A, B)
    #print(bHat)
    hmmCountModel = {}
        
    with open("trainmodel.dat", "wb") as outFile:
        #outFile.write("<A>")

        hmmCountModel["aMatrix"] = aHat
        #hmmCountModel["bMatrix"] = bHat
        pickle.dump(hmmCountModel, outFile)


        print("Training completed. Saving to model.dat")

main()


