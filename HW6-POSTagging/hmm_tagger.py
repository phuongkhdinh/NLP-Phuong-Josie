# hmm_tagger.py
# By Phuong Dinh & Josie Bealle
import sys
import re
import pickle
import math

def Viterbi(obs, stateGraph, aMatrix, bMatrix):

	N = len(stateGraph)
	T = len(obs)
	print(T)
	#viterbi = {}
	#backpointer = {}
	viterbi = [[None for col in range(T)] for row in range(N+1)]	 #Matrix [N+2, T]
	backpointer = [[0 for col in range(T)] for row in range(N+1)]	
	#aMatrix in form {(qi, qj): double prob} given tag i, prob of tagj
	#bMatrix in form {(w, t): double prob} given tag t, prob of w
	#initialization
	for state in range(0, N): #MIGHT be N-1 or 1->N

		if (('START',stateGraph[state]) in aMatrix) and ((obs[0], stateGraph[state]) in bMatrix):
		#try:
			#viterbi[(stateGraph[state], 0)] = math.log(aMatrix[("START",stateGraph[state])]) + math.log(bMatrix[(obs[0], stateGraph[state])])
			viterbi[state][0] = math.log(aMatrix[("START",stateGraph[state])]) + math.log(bMatrix[(obs[0], stateGraph[state])])
		#except:
		elif ('START',stateGraph[state]) in aMatrix:
		 	viterbi[state][0] = None
		else:
			print('START'+stateGraph[state])
		backpointer[state][0] = [stateGraph[state]]
	#Recursion
	for time in range(1,T): #MIGHT BE T-1
		for state in range(0,N):
			for statePrime in range(0,N):
				#if (obs[time],stateGraph[state]) in bMatrix: #and (stateGraph[statePrime], stateGraph[state]) in aMatrix:
				if viterbi[statePrime][time-1] != None and (obs[time],stateGraph[state]) in bMatrix:
					calVi = viterbi[statePrime][time-1] + \
					math.log(aMatrix[(stateGraph[statePrime], stateGraph[state])]) + math.log(bMatrix[(obs[time],stateGraph[state])])
					print(calVi, stateGraph[state], time)
					if viterbi[state][time] == None or calVi > viterbi[state][time]:
						viterbi[state][time] = calVi
						backpointer[state][time] = backpointer[statePrime][time-1] + [stateGraph[state]]

				else:
					print("Who: "+ stateGraph[statePrime]+"obs"+obs[time])
	# Termination
	#viterbi[N][T] = 0
	for state in range(0,N):
		if (stateGraph[state], "END") in aMatrix and viterbi[state][T-1] != None:
			calEndVit = viterbi[state][T-1] + math.log(aMatrix[(stateGraph[state], "END")])

			if viterbi[N][T-1] == None or calEndVit > viterbi[N][T-1]:
				print(calEndVit)
				print(stateGraph[state])
				viterbi[N][T-1] = calEndVit
				backpointer[N][T-1] = backpointer[state][T-1]

	return backpointer[N][T-1]

def main():

	modelFile = sys.argv[1]
	for line in sys.stdin:
		obs = line.strip("\n").split(" ")

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
	stateGraph.remove("START")
	stateGraph.remove("END")
	backpointerPath = Viterbi(obs, stateGraph, aMatrix, bMatrix)
	print(backpointerPath)

main()
