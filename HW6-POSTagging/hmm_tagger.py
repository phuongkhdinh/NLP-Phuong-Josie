# hmm_tagger.py
# By Phuong Dinh & Josie Bealle
import sys
import re
import pickle

def Viterbi(obs, stateGraph, aMatrix, bMatrix):
	viterbi = {} #Matrix [N+2, T]
	backpointer = {}
	#aMatrix in form {(qi, qj): double prob} given tag i, prob of tagj
	#bMatrix in form {(w, t): double prob} given tag t, prob of w
	N = len(stateGraph)
	#print(stateGraph)
	T = len(obs)

	#initialization
	for state in range(0, N): #MIGHT be N-1 or 1->N

		if (("START",stateGraph[state]) in aMatrix) and ((obs[0], stateGraph[state]) in bMatrix):
			viterbi[(stateGraph[state], 0)] = aMatrix[("START",stateGraph[state])] * bMatrix[(obs[0], stateGraph[state])]#MIGHT USE LOG
		else:

			viterbi[(stateGraph[state], 0)] = 0
		backpointer[(stateGraph[state], 0)] = 0
	#Recursion
	for time in range(1,T): #MIGHT BE T-1
		for state in range(0,N):
			viterbi[(stateGraph[state], time)] = 0
			for statePrime in range(0,N):
				if (obs[time],stateGraph[state]) in bMatrix and (stateGraph[statePrime], stateGraph[state]) in aMatrix:
					calVi = viterbi[(stateGraph[statePrime],time-1)] * \
					aMatrix[(stateGraph[statePrime], stateGraph[state])] * bMatrix[(obs[time],stateGraph[state])] 
					if calVi > viterbi[stateGraph[state], time]:
						viterbi[(stateGraph[state], time)] = calVi
						backpointer[(stateGraph[state], time)] = stateGraph[statePrime]
	# Termination
	viterbi["END",T] = 0
	print(viterbi)
	for state in range(0,N):
		if (stateGraph[state], "END") in aMatrix:
			calEndVit = viterbi[(stateGraph[state],T)] * aMatrix[(stateGraph[state], "END")]
			if calEndVit > viterbi[("END",T)]:
				viterbi[("END",T)] = calEndVit
			backpointer[("END",T)] = stateGraph[state]

	return backpointer

def main():

	modelFile = sys.argv[1]
	for line in sys.stdin:
		obs = line.split(" ")

	with open('countmodel.dat', 'rb') as handle:
		matrixes = pickle.loads(handle.read())
	aMatrix = matrixes["aMatrix"]
	bMatrix = matrixes["bMatrix"]


	stateGraph = []
	for state in aMatrix:
		if state[0] not in stateGraph:
			stateGraph.append(state[0])
		if state[1] not in stateGraph:
			stateGraph.append(state[1])
	backpointerPath = Viterbi(obs, stateGraph, aMatrix, bMatrix)
	print(backpointerPath)

main()
