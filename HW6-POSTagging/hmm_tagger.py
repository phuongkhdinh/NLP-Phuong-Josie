# hmm_tagger.py
# By Phuong Dinh & Josie Bealle
import sys
import re
import pickle
import math

def Viterbi(obs, stateGraph, aMatrix, bMatrix):

	N = len(stateGraph)
	T = len(obs)
	#viterbi = {}
	#backpointer = {}
	viterbi = [[None for col in range(T)] for row in range(N+1)]	 #Matrix [N+2, T]
	backpointer = [[0 for col in range(T)] for row in range(N+1)]	
	#aMatrix in form {(qi, qj): double prob} given tag i, prob of tagj
	#bMatrix in form {(w, t): double prob} given tag t, prob of w
	#initialization
	for state in range(0, N): #MIGHT be N-1 or 1->N

		if bMatrix[obs[0], stateGraph[state]] == 0: #We have the first word with this tag
			firstWord = "<UNK>"
		#try:
			#viterbi[(stateGraph[state], 0)] = math.log(aMatrix[("START",stateGraph[state])]) + math.log(bMatrix[(obs[0], stateGraph[state])])
			viterbi[state][0] = aMatrix[("START",stateGraph[state])] + bMatrix[(firstWord, stateGraph[state])] - 10
		#except:
		else:

			viterbi[state][0] = aMatrix[("START",stateGraph[state])] + bMatrix[(obs[0], stateGraph[state])]
		#elif ('START',stateGraph[state]) in aMatrix:
		# 	viterbi[state][0] = aMatrix[("START",stateGraph[state])] - 99

		backpointer[state][0] = [stateGraph[state]]
	#Recursion
	for time in range(1,T): #MIGHT BE T-1
		for state in range(0,N):
			for statePrime in range(0,N):
				#if (obs[time],stateGraph[state]) in bMatrix: #and (stateGraph[statePrime], stateGraph[state]) in aMatrix:

				if viterbi[statePrime][time-1] != None:
					if bMatrix[obs[time], stateGraph[state]] == 0: #We have the first word with this tag
						word = "<UNK>"
						calVi = viterbi[statePrime][time-1] + \
						aMatrix[(stateGraph[statePrime], stateGraph[state])] + bMatrix[(word,stateGraph[state])] - 10
					else:
						calVi = viterbi[statePrime][time-1] + \
						aMatrix[(stateGraph[statePrime], stateGraph[state])] + bMatrix[(obs[time],stateGraph[state])]
					if viterbi[state][time] == None or calVi > viterbi[state][time]:
						viterbi[state][time] = calVi
						backpointer[state][time] = backpointer[statePrime][time-1] + [stateGraph[state]]

	# Termination
	#viterbi[N][T] = 0
	for state in range(0,N):
		if (stateGraph[state], "END") in aMatrix and viterbi[state][T-1] != None:
			calEndVit = viterbi[state][T-1] + aMatrix[(stateGraph[state], "END")]

			if viterbi[N][T-1] == None or calEndVit > viterbi[N][T-1]:
				viterbi[N][T-1] = calEndVit
				backpointer[N][T-1] = backpointer[state][T-1]
	return backpointer[N][T-1]

def main():

	modelFile = sys.argv[1]
	for line in sys.stdin:
		obs = line.lower().strip("\n").split(" ")

	with open('countmodel.dat', 'rb') as handle:
		matrixes = pickle.loads(handle.read())
	aMatrix = matrixes["aMatrix"]
	bMatrix = matrixes["bMatrix"]
	print(aMatrix)

	stateGraph = []
	for state in aMatrix:
		if state[0] not in stateGraph:
			stateGraph.append(state[0])
		if state[1] not in stateGraph:
			stateGraph.append(state[1])
	stateGraph.remove("START")
	stateGraph.remove("END")
	backpointerPath = Viterbi(obs, stateGraph, aMatrix, bMatrix)
	string = ""
	originalObs = line.strip("\n").split(" ")
	for i in range(len(originalObs)):
		string = string + originalObs[i] + "/" + backpointerPath[i] + " "
	string = string + "\n"

	sys.stdout.write(string)

main()
