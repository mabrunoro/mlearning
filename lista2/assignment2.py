#!/usr/bin/env python3

import numpy as np
import random
import scipy.stats as stats
import math
import sys

def autov(X):
	return np.linalg.eig(np.cov(X,rowvar=False))

def mean(X):
	return np.mean(np.array(X), axis=0)

def standard(X):
	return np.std(np.array(X), axis=0)

def bin(intervalos, amostra):
	res = np.zeros((len(intervalos),len(intervalos[-1])))
	for i in range(amostra.shape[0]):
		for j in range(intervalos[i].shape[0]):
			if(intervalos[i][j] <= amostra[i]):
				res[i,j] += 1
				break
	return res


def exe01(arg):
	print("\n\nExercício 1")
	data = []
	with open(arg) as f:
		for l in f:
			data.append(l.rstrip().split(','))

	data = np.array(data)

	# A
	print("\nLetra A")
	res = np.unique(data[:,0] == 'med', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x1=med) =", 100*res[1][0]/data.shape[0])
		else:
			print("P(x1=med) =", 100*res[1][1]/data.shape[0])
	else:
		if(res[0][0]):
			print("P(x1=med) =", 100*res[1][0]/data.shape[0])
		else:
			print("P(x1=med) =", 0)

	res = np.unique(data[:,1] == 'low', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x2=low) =", 100*res[1][0]/data.shape[0])
		else:
			print("P(x2=low) =", 100*res[1][1]/data.shape[0])
	else:
		if(res[0][0]):
			print("P(x2=low) =", 100*res[1][0]/data.shape[0])
		else:
			print("P(x2=low) =", 0)

	# B
	print("\nLetra B")
	aux = data[data[:, 2]=='2']
	res = np.unique(aux[:, 5] == 'high', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x6=high | x3=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x6=high | x3=2) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x6=high | x3=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x6=high | x3=2) =", 0)

	aux = data[data[:, 3]=='4']
	res = np.unique(aux[:, 1] == 'low', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x2=low | x4=4) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x2=low | x4=4) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x2=low | x4=4) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x2=low | x4=4) =", 0)

	# C
	print("\nLetra C")
	aux = data[data[:, 1] == 'low']
	aux = aux[aux[:, 4] == 'small']
	res = np.unique(aux[:, 0] == 'low', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x1=low | x2=low, x5=small) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x1=low | x2=low, x5=small) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x1=low | x2=low, x5=small) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x1=low | x2=low, x5=small) =", 0)

	aux = data[data[:, 0] == 'med']
	aux = aux[aux[:, 2] == '2']
	res = np.unique(aux[:, 3] == '4', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x4=4 | x1=med, x3=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x4=4 | x1=med, x3=2) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x4=4 | x1=med, x3=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x4=4 | x1=med, x3=2) =", 0)

	# C
	print("\nLetra D")
	aux = data[data[:, 2] == '2']
	aux = aux[aux[:, 3] == '2']
	res = np.unique(aux[:, 1] == 'vhigh', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x2=vhigh | x3=2, x4=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x2=vhigh | x3=2, x4=2) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x2=vhigh | x3=2, x4=2) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x2=vhigh | x3=2, x4=2) =", 0)

	aux = data[data[:, 4] == 'med']
	aux = aux[aux[:, 0] == 'med']
	res = np.unique(aux[:, 2] == '4', return_counts=True)
	if(res[0].shape[0] > 1):
		if(res[0][0]):
			print("P(x3=4 | x5=med, x1=med) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x3=4 | x5=med, x1=med) =", 100*res[1][1]/aux.shape[0])
	else:
		if(res[0][0]):
			print("P(x3=4 | x5=med, x1=med) =", 100*res[1][0]/aux.shape[0])
		else:
			print("P(x3=4 | x5=med, x1=med) =", 0)


def exe02(arg):
	print("\n\nExercício 2")
	data = []
	classes = []
	with open(arg) as f:
		for l in f:
			e = l.rstrip().split(',')
			classes.append(e[0])
			data.append(list(map(int, e[1:])))

	classes = np.array(classes)
	data = np.array(data)

	# A
	print("\nLetra A")
	acc = []
	for tempo in range(10):
		amostras = random.sample(range(data.shape[0]), data.shape[0])
		treino = amostras[ : math.floor(data.shape[0] * 0.75)]
		teste = amostras[math.floor(data.shape[0] * 0.75) : ]

		uc = np.unique(classes)	# unique classes
		up = []	# probabilidade de cada classe (treino)	- P(c)
		um = []	# média de cada classe (treino)
		us = []	# desvio padrão de cada classe (treino)
		ic = classes[treino]

		for i in uc:
			res = np.unique(ic == i, return_counts=True)
			if(res[0][0]):
				up.append(res[1][0]/ic.shape[0])
			elif(res[0].shape[0] == 1):
				up.append(0)
			else:
				up.append(res[1][1]/ic.shape[0])

			id = data[treino][ic == i]	# seleciona os dados de cada classe (treino)
			um.append(mean(id))
			us.append(standard(id))

		toc = 0	# total de corretos
		for i in teste:
			best = [0, up[0]]
			for k in stats.norm(um[0], us[0]).pdf(data[i]):
				best[1] *= k

			for j in range(1, uc.shape[0]):
				aux = up[j]
				for k in stats.norm(um[j], us[j]).pdf(data[i]):
					aux *= k
				if(aux > best[1]):
					best[1] = aux
					best[0] = j

			if(classes[i] == uc[best[0]]):	# confere se a classe mais provável (maior probabilidade) corresponde à classe da amostra
				toc += 1

		acc.append(toc / len(teste))
	# print(acc)
	print("Acurácia média:", np.mean(acc))
	print("Desvio padrão da acurácia:", np.std(acc))

	# B
	print("\nLetra B")
	acc = []
	for tempo in range(10):
		amostras = random.sample(range(data.shape[0]), data.shape[0])
		treino = amostras[ : math.floor(data.shape[0] * 0.75)]
		teste = amostras[math.floor(data.shape[0] * 0.75) : ]

		uc = np.unique(classes)	# unique classes
		uinter = []	# limites de intervalo de cada classe (treino)
		uqinter = []	# quantidade de elementos de cada classe nos intervalos
		ic = classes[treino]

		for i in uc:
			id = data[treino][ic == i]	# seleciona os dados de cada classe (treino)
			umax = np.amax(id, axis=0)
			umin = np.amin(id, axis=0)

			uinter.append([])
			for j in range(umax.shape[0]):
				uinter[-1].append(np.flip(np.linspace(umin[j], umax[j], 5, endpoint=False), axis=0))
			uqinter.append(np.zeros((len(uinter[-1]),len(uinter[-1][0]))))

			for j in id:
				uqinter[-1] += bin(uinter[-1], j)
			uqinter[-1] /= id.shape[0]

		toc = 0	# total de corretos
		for i in teste:
			best = [-1,-1]
			for j in range(len(uqinter)):
				aux = 1
				for k in np.sum(uqinter[j]*bin(uinter[j], data[i]), axis=1):
					aux *= k
				if(best[1] < aux):
					best[0] = j
					best[1] = aux

			if(classes[i] == uc[best[0]]):	# confere se a classe mais provável (maior probabilidade) corresponde à classe da amostra
				toc += 1

		acc.append(toc / len(teste))
	# print(acc)
	print("Acurácia média:", np.mean(acc))
	print("Desvio padrão da acurácia:", np.std(acc))

	# C
	print("\nLetra C")
	acc = []
	for tempo in range(10):
		amostras = random.sample(range(data.shape[0]), data.shape[0])
		treino = amostras[ : math.floor(data.shape[0] * 0.75)]
		teste = amostras[math.floor(data.shape[0] * 0.75) : ]

		uc = np.unique(classes)	# unique classes
		uinter = []	# limites de intervalo de cada classe (treino)
		uqinter = []	# quantidade de elementos de cada classe nos intervalos
		ic = classes[treino]

		for i in uc:
			id = data[treino][ic == i]	# seleciona os dados de cada classe (treino)
			umax = np.amax(id, axis=0)
			umin = np.amin(id, axis=0)

			uinter.append([])
			for j in range(umax.shape[0]):
				uinter[-1].append(np.flip(np.linspace(umin[j], umax[j], 5, endpoint=False), axis=0))
			uqinter.append(np.zeros((len(uinter[-1]),len(uinter[-1][0]))))

			for j in id:
				uqinter[-1] += bin(uinter[-1], j)
			uqinter[-1] = (uqinter[-1] + 1)/ (id.shape[0] + id.shape[1])

		toc = 0	# total de corretos
		for i in teste:
			best = [-1,-1]
			for j in range(len(uqinter)):
				aux = 1
				for k in np.sum(uqinter[j]*bin(uinter[j], data[i]), axis=1):
					aux *= k
				if(best[1] < aux):
					best[0] = j
					best[1] = aux

			if(classes[i] == uc[best[0]]):	# confere se a classe mais provável (maior probabilidade) corresponde à classe da amostra
				toc += 1

		acc.append(toc / len(teste))
	# print(acc)
	print("Acurácia média:", np.mean(acc))
	print("Desvio padrão da acurácia:", np.std(acc))


def exe03():
	print("\n\nExercício 3")

	# A
	print("\nLetra A")
	print("A rede 2 representa corretamente a estrutura das informações.")
	print("A rede 3 também está correta, se levarmos em consideração a ordem dos dados.")

	# B
	print("\nLetra B")
	print("A rede 2 representa melhor a informação da questão, pois é necessário menos parâmetros do que a rede 3 (para sabermos M1, precisamos de F1 e F2 - ou N e M2).")


def exe04():
	print("\n\nExercício 4")

	# A
	print("\nLetra A")
	print("A e B são independentes de C caso F não seja conhecido, ou seja, F cria uma dependência de A e B com C.")

	# B
	print("\nLetra B")
	print("Errado. B depende de C quando F é conhecido (F desbloqueia C).")

	# C
	print("\nLetra C")
	print("G é dependente de E por conta de D, mesmo que desconhecer I não crie dependência entre G e E.")

	# D
	print("\nLetra D")
	print("C depende de E pois a observação de I cria dependência com H.")

	# E
	print("\nLetra E")
	print("D é independente de H enquanto I não for observado.")

	# F
	print("\nLetra F")
	print("H depende de D por conta da observação de I (há dependência com G).")

	# G
	print("\nLetra G")
	print("I é d-separado de B por conta de F e, consequentemente, independente de A.")

	# H
	print("\nLetra H")
	print("F é independente de D pois C bloqueia, mesmo que pudesse haver dependência por conta da observação de G.")


def exe05(arg):
	print("\n\nExercício 5")
	data = []
	with open(arg) as f:
		for l in f:
			e = l.rstrip().split()
			if(e[0] == 'C'):
				continue
			data.append(list(map(int, e)))

	data = np.array(data)

	# A
	print("\nLetra A")

	# Pc
	Pc = np.sum(data[:, 0]) / data.shape[0]

	# Pf
	Pf = np.sum(data[:, 1]) / data.shape[0]

	# Pe
	Pe = np.array([[0.0, 0.0], [0.0, 0.0]])
	for i in [0, 1]:
		for j in [0, 1]:
			aux = np.logical_and((data[:,0] == i), (data[:,1] == j))
			Pe[i, j] = np.sum(data[aux,2]) / aux.shape[0]

	# Pr
	Pr = np.array([0.0, 0.0])
	for i in [0, 1]:
		aux = data[:,0] == i
		Pr[i] = np.sum(data[aux,3]) / aux.shape[0]

	# Pa
	Pa = np.array([[0.0, 0.0], [0.0, 0.0]])
	for i in [0,1]:
		for j in [0,1]:
			aux = np.logical_and((data[:,3] == i), (data[:,2] == j))
			Pa[i, j] = np.sum(data[aux,4]) / aux.shape[0]

	print("As probabilidades seguem.\nAs matrizes seguem a ordem da rede, ou seja, a matriz acidente tem a forma P[Ruas alagadas, Engarrafamento].")
	print("\nPc:")
	print(Pc)
	print("\nPf:")
	print(Pf)
	print("\nPe:")
	print(Pe)
	print("\nPr:")
	print(Pr)
	print("\nPa:")
	print(Pa)

	# B
	print("\nLetra B")
	print("A probabilidade é Pa[0,1] + Pa[1,1] =", Pa[0,1] + Pa[1,1])

	# C
	print("\nLetra C")
	print("A probabilidade é 1-Pa[1,1] =", 1-Pa[1,1])

	# D
	print("\nLetra D")
	print("A probabilidade é calculada da forma:")
	print("\tP(E=1 | A=1) = P(E=1, A=1) / P(A)")
	print("\tP(E=1, A=1) = P(A | E=1, R=x) P(R=x | C=y) P(E=1 | C=y, F=z) P(C=y, F=z)")
	print("\tsendo que 'x', 'y' e 'z' assumem os valores 0 e 1, independentemente.")




def main(arg=None):
	if(arg is None):
		folder = "bases/"
	else:
		folder = arg

	# exe01(folder + "car.data.txt")
	# exe02(folder + "balance-scale.data.txt")
	# exe03()
	# exe04()
	exe05("transito.txt")

if __name__ == '__main__':
	main("bases/")
