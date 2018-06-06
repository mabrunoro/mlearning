#!/usr/bin/env python3
# Primeira Lista de Exercícios de Aprendizado de Máquinas
# Aluno: Marco Aurélio Brunoro Thomé

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import scipy.stats as st
import math
import tsne
import random

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

# def readfile(file, mode=None):
# 	data = []
# 	with open(file) as f:
# 		for i in f:
# 			if(mode is None):
# 				data.append(list(map(int,i.split())))
# 			else:
# 				print(i)
# 	return np.array(data)

def autov(X):
	return np.linalg.eig(np.cov(X,rowvar=False))

def mean(X):
	return np.mean(np.array(X), axis=0)

def standard(X):
	return np.std(np.array(X), axis=0)

def variance(X):
	return np.var(np.array(X), axis=0)

def center(X, corr=False):
	if(corr):
		return (X - mean(X)) / variance(X)
	else:
		return (X - mean(X))

def decorr(X, PHI=None):
	if(PHI is None):
		_, PHI = autov(X)
	return np.dot(X, PHI)

def projo(X, vec):
	return np.dot(X, np.transpose(vec))

def nn(clss,attr,ntreino):
	clssa = clss.copy()
	for i in range(ntreino,attr.shape[0]):	# para cada item de teste
		distancia = 0
		for j in range(ntreino):	# passa por cada item de treino
			if(j == 0):
				distancia = np.linalg.norm(attr[i,2:] - attr[j,2:])
				clssa[i] = clss[j]
			else:
				aux = np.linalg.norm(attr[i,2:] - attr[j,2:])
				if(aux < distancia):
					distancia = aux
					clssa[i] = clss[j]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)

	if(res[0][0]):
		print('\nAcurácia NN:', 100*res[1][0]/(clssa.shape[0]-ntreino))
	else:
		print('\nAcurácia NN:', 100*res[1][1]/(clssa.shape[0]-ntreino))

def rochio(clss, attr, ntreino):
	classes = np.unique(clss)
	clssa = clss.copy()
	medias = []
	for i in classes:
		medias.append(mean(attr[:ntreino][clss[:ntreino] == i]))
	for i in range(ntreino,clss.shape[0]):
		distancia = np.linalg.norm(attr[i] - medias[0])
		clssa[i] = classes[0]
		for j in range(1,len(medias)):
			aux = np.linalg.norm(attr[i] - medias[j])
			if(aux < distancia):
				distancia = aux
				clssa[i] = clss[j]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)

	if(res[0][0]):
		print('\nAcurácia Rochio:', 100*res[1][0]/(clssa.shape[0]-ntreino))
	else:
		print('\nAcurácia Rochio:', 100*res[1][1]/(clssa.shape[0]-ntreino))

def bins(vec, size=4):
	vec = np.array(vec)
	for i in range(vec.shape[1]):
		args = vec[:,i].argsort()
		size2 = size
		for i in range(0,vec.shape[0],size2):
			if(i + size2 > vec.shape[0]):
				size2 = (vec.shape[0]%size2)
			sum = 0
			for j in range(size2):
				sum += vec[args[i+j]]
			sum /= size2
			for j in range(size2):
				vec[args[i+j]] = sum

def getw(x, y, ord=1):
	# val = [x y]
	X = np.ones(x.shape[0])
	if(len(x.shape) == 1):	# uma entrada e uma saída
		for i in range(ord):
			X = np.vstack((X, x**(i+1)))
		# lembrete: a variável X é na verdade a transposta do X dos slides
	else:
		X = np.vstack((X, x.T))
	return np.dot(np.dot(np.linalg.inv(np.dot(X,X.T)),X),y)

def kendall(x, y, sig=0.05):
	# vec = [x y]
	if((x.shape[0] < 2) or (y.shape[0] != x.shape[0])):
		print('Erro Kendall. Número de exemplos.')
	else:
		N = x.shape[0]
		sum = 0
		nx = 0
		ny = 0
		for i in range(1,N):
			for j in range(i):
				sum += np.sign(x[j] - x[i])*np.sign(y[j] - y[i])
				if((x[j] - x[i]) != 0):
					nx += 1
				if((y[j] - y[i]) != 0):
					ny += 1
		if((nx == 0) or (ny == 0)):
			print('Erro Kendall. Somente pares empatados.')
		else:
			tau = sum / math.sqrt(nx*ny)
			z = st.norm.ppf(1-(sig/2))
			if(abs(tau) > (z * math.sqrt( (2*(2*N+5)) / (9*N*(N-1)) ))):
				print("Kendall: hipótese nula rejeitada, há possibilidade de haver dependência entre 'x' e 'y' com",1-sig,"de significância.\nTau =",tau)
			else:
				print("Kendall: hipótese nula aceita, não há possibilidade de haver dependência entre 'x' e 'y' com",1-sig,"de significância.\nTau =",tau)

def pearson(x, y, sig, N=20):
	# N = vec.shape[0]
	p = np.cov(x,y,rowvar=False)[0,1]/math.sqrt(variance(x)*variance(y))
	if(p == 1):
		print("Pearson: correlação linear perfeita positiva entre 'x' e 'y'.")
	elif(p == -1):
		print("Pearson: correlação linear perfeita negativa entre 'x' e 'y'.")
	elif(p == 0):
		print("Pearson: hipótese nula. 'x' e 'y' são descorrelacionados.")
	amostras = random.sample(range(0, x.shape[0]), N)
	print('\nUsando',N,'amostras...')
	p = np.cov(x[amostras], y[amostras], rowvar=False)[0,1] / math.sqrt(variance(x[amostras]) * variance(y[amostras]))
	to = p*math.sqrt(N-1)/math.sqrt(1-p*p)
	t = st.t.ppf(sig/2,N-2)
	if(abs(to) > t):
		print("Pearson: hipótese nula rejeitada. Existe a possibilidade de haver dependência entre 'x' e 'y' com",sig,'de significância.','\nto =',to,'\nt =',t)
	else:
		print('Pearson: hipótese nula aceita para',sig,'de significância.\nto =',to,'\nt =',t)

def snedecor(x, y, w, Fs=3.908):
	X = np.hstack((np.array([ [1] * x.shape[0] ]).T, x))
	n = w.shape[0]
	N = x.shape[0]
	q = n - 1
	wres = w.copy()

	rss = ((y - np.dot(X,w)) ** 2).sum()
	# print('RSS:',rss)
	for i in range(x.shape[1]):
		waux = w.copy()
		waux[i] = 0
		rssi = ((y - np.dot(X,waux)) ** 2).sum()
		f = ((rssi - rss)/(n - q))/(rss/(N - n - 1))
		# print('\nCoeficiente:',i)
		# print('RSSi:',rssi)
		# print('F:',f)
		if(f <= Fs):
			print('\tcoeficiente w',i,' pode ser desconsiderado',sep='')
			wres[i] = 0
	return wres




# Parte 1
def exe1(folder='bases/'):
	print('Exercício 1')
	# data = readfile(folder + 'CNAE_9_reduzido.txt')
	data = []
	with open(folder + 'CNAE_9_reduzido.txt') as f:
		for i in f:
			data.append(list(map(int,i.split())))
	data = np.array(data)

	rotulos = np.array(list(map(int, data[:,0])))
	X = data[:,1:]

	# A
	print('\nLetra A')
	X = center(X, corr=False)

	autovalores,autovetores = autov(X)

	argmax = np.argpartition(-autovalores, 2)[:2]

	X = np.dot(X, autovetores[:,[argmax[0],argmax[1]]])
	print(autovetores[:,[argmax[0],argmax[1]]])

	for (i,color) in zip(rotulos,colors[:max(rotulos)]):
		idx = np.where(rotulos == i)
		plt.scatter(X[idx,0], X[idx,1], c=color, label=rotulos[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
	# plt.ylim(math.floor(min(X[:,argmax[1]])*1.2), math.ceil(max(X[:,argmax[1]])*1.2))
	plt.legend()
	plt.show()

	# B
	print('\nLetra B')
	W = np.diag(autovalores[[argmax[0],argmax[1]]]**(-0.5))
	X = np.dot(X, W)

	for (i,color) in zip(rotulos,colors[:max(rotulos)]):
		idx = np.where(rotulos == i)
		plt.scatter(X[idx,argmax[0]], X[idx,argmax[1]], c=color, label=rotulos[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
	# plt.ylim(math.floor(min(X[:,argmax[1]])*1.2), math.ceil(max(X[:,argmax[1]])*1.2))
	plt.legend()
	plt.show()

	# C
	print('\nLetra C')
	Y = tsne.tsne(data[:,1:], 2, 20)

	for (i,color) in zip(rotulos,colors[:max(rotulos)]):
		idx = np.where(rotulos == i)
		plt.scatter(Y[idx,0], Y[idx,1], c=color, label=rotulos[i], cmap=plt.cm.Paired, edgecolor='black', s=20)
	plt.ylim(math.floor(min(Y[:,1])*1.2), math.ceil(max(Y[:,1])*1.2))
	plt.legend()
	plt.show()

	# D
	print('\nLetra D')
	treino = data[:480,:]
	teste = np.hstack((np.array([data[480:,:].shape[0]*[0]]).T,data[480:,1:]))
	for i in teste:
		for j in range(treino.shape[0]):
			distancia = 0
			if(j == 0):
				distancia = np.linalg.norm(i[1:]-treino[j,1:])
				i[0] = treino[j,0]
			else:
				aux = np.linalg.norm(i[1:]-treino[j,1:])
				if(aux < distancia):
					distancia = aux
					i[0] = treino[j,0]
	res = np.unique(data[480:,0] == teste[:,0], return_counts=True)
	if(res[0][0]):
		print('Acurácia:', 100*res[1][0]/teste.shape[0])
	else:
		print('Acurácia:', 100*res[1][1]/teste.shape[0])
	# a acurácia é baixa, o que significa que não é fácil classificar os elementos, por isso, o gráfico que mais se aproxima da realidade é o gerado a partir do PCA




def exe2(folder='bases/'):
	print('Exercício 2')
	# data = readfile(folder + 'wdbc.data.txt', True)
	data = []
	with open(folder + 'wdbc.data.txt') as f:
		for i in f:
			for j in i.split():
				data.append(j.split(','))
	data = np.array(data)
	clss = np.array(list(map(str,data[:,1])))
	attr = []
	for i in range(data.shape[0]):
		attr.append(list(map(float, data[i,2:])))
	attr = np.array(attr)

	# A
	print('\nLetra A')
	# a primeira coluna é o ID do paciente, que pode ser descartado
	# a segunda coluna indica se o câncer é maligno ou benigno

	nn(clss, attr, 300)

	# ntreino = 300
	# clssa = clss.copy()
	# for i in range(ntreino,data.shape[0]):	# para cada item de teste
	# 	distancia = 0
	# 	for j in range(ntreino):	# passa por cada item de treino
	# 		if(j == 0):
	# 			distancia = np.linalg.norm(attr[i,2:] - attr[j,2:])
	# 			clssa[i] = clss[j]
	# 		else:
	# 			aux = np.linalg.norm(attr[i,2:] - attr[j,2:])
	# 			if(aux < distancia):
	# 				distancia = aux
	# 				clssa[i] = clss[j]
	# res = np.unique(data[ntreino:,1] == clssa[ntreino:], return_counts=True)
	# if(res[0][0]):
	# 	print('\nAcurácia:', 100*res[1][0]/(clssa.shape[0]-ntreino))
	# else:
	# 	print('\nAcurácia:', 100*res[1][1]/(clssa.shape[0]-ntreino))

	# B
	print('\nLetra B')
	X = attr
	X = center(X, corr=True)
	autovalores,autovetores = autov(X)
	print('\nAutovalores:\n',autovalores,sep='')
	tvar = autovalores.sum()	# variância total
	pvar = 0	# variância parcial (deve ser meior que 90%)
	argmax = np.argpartition(-autovalores, autovalores.shape[0]-1)
	j = 0
	for i in argmax:
		if(pvar > 0.9):
			break
		else:
			j += 1
			pvar += autovalores[i]/tvar
	print('\nNúmero de componentes selecionados para variância de 0.9+:', j)

	X = np.dot(X, autovetores[:,argmax[:j]])

	nn(clss, X, 300)

	# ntreino = 300
	# clssb = clss.copy()
	# for i in range(ntreino,X.shape[0]):	# para cada item de teste
	# 	distancia = 0
	# 	for j in range(ntreino):	# passa por cada item de treino
	# 		if(j == 0):
	# 			distancia = np.linalg.norm(X[i,2:] - X[j,2:])
	# 			clssb[i] = clss[j]
	# 		else:
	# 			aux = np.linalg.norm(X[i,2:] - X[j,2:])
	# 			if(aux < distancia):
	# 				distancia = aux
	# 				clssb[i] = clss[j]
	# res = np.unique(data[ntreino:,1] == clssb[ntreino:], return_counts=True)
	# if(res[0][0]):
	# 	print('\nAcurácia:', 100*res[1][0]/(clssb.shape[0]-ntreino))
	# else:
	# 	print('\nAcurácia:', 100*res[1][1]/(clssb.shape[0]-ntreino))

	# C
	print('\nLetra C')
	classes = np.unique(clss, return_counts=True)
	cmean = np.array([mean(attr[clss == i]) for i in classes[0]])
	tmean = mean(attr)

	Sw = np.zeros((cmean.shape[1],cmean.shape[1]))
	for i in range(attr.shape[0]):
		aux = attr[i] - cmean[classes[0] == clss[i]][0]
		Sw += np.outer(aux,aux)

	Sb = np.zeros((cmean.shape[1],cmean.shape[1]))
	for i in range(len(classes[1])):
		aux = cmean[i] - tmean
		Sb += np.outer(aux,aux) * classes[1][i]

	autovalores,autovetores = linalg.eig(Sb,Sw)
	minimo = np.argmin(autovalores)
	autovalores = np.delete(autovalores, minimo)
	autovetores = np.delete(autovetores, minimo, 0)

	Y = np.dot(autovetores, attr.T).T	# projeção

	nn(clss,Y,300)

	# ntreino = 300
	# clssc = clss.copy()
	# for i in range(ntreino,Y.shape[0]):	# para cada item de teste
	# 	distancia = 0
	# 	for j in range(ntreino):	# passa por cada item de treino
	# 		if(j == 0):
	# 			distancia = np.linalg.norm(Y[i,2:] - Y[j,2:])
	# 			clssc[i] = clss[j]
	# 		else:
	# 			aux = np.linalg.norm(Y[i,2:] - Y[j,2:])
	# 			if(aux < distancia):
	# 				distancia = aux
	# 				clssc[i] = clss[j]
	# res = np.unique(data[ntreino:,1] == clssc[ntreino:], return_counts=True)
	# if(res[0][0]):
	# 	print('\nAcurácia:', 100*res[1][0]/(clssc.shape[0]-ntreino))
	# else:
	# 	print('\nAcurácia:', 100*res[1][1]/(clssc.shape[0]-ntreino))
	# esta abordagem foca na classificação dos exemplos; o vetor de maior de variância não é necessariamente o que gera a melhor projeção para classificar os dados




def exe3(folder='bases/'):
	print('Exercício 3')
	data = []
	with open(folder + 'nebulosa_train.txt') as f:
		for i in f:
			data.append(i.rstrip().split())

	ntreino = len(data)
	with open(folder + 'nebulosa_test.txt') as f:
		for i in f:
			data.append(i.rstrip().split())

	# data = np.array(data)

	# A
	print('\nLetra A')
	data2 = []
	for i in data:
		if('?' not in i):
			data2.append([float(j) for j in i])

	data2 = np.array(data2)
	media = mean(data2)

	for i in data:
		for j in range(len(i)):
			if(i[j] == '?'):
				i[j] = media[j]
			else:
				i[j] = float(i[j])

	# data = np.array(data)[:,2:]	# retira os IDs e nomes
	data = np.array(data)
	clss = data[:,-1]	# classes
	attr = data[:,2:-1]	# atributos

	nn(clss, attr, ntreino)
	rochio(clss, attr, ntreino)

	# B
	print('\nLetra B')
	# reduzir os ruídos, as redundâncias, inconsistências, outliers e a interferência dos dados incompletos
	# remover dados redundantes (usando ID)
	data2 = np.unique(data, axis=0)
	# exemplos incompletos já tiveram a média inserida no espaço de dados faltantes
	clssb = data2[:,-1]	# classes
	attrb = data2[:,2:-1]	# atributos
	unicos = np.unique(clssb, return_counts=True)

	# removendo classes com apenas um exemplo
	unicos = unicos[0][unicos[1] == 1]
	for i in unicos:
		aux = clssb != i
		clssb = clssb[aux]
		attrb = attrb[aux]

	# removendo outliers
	desvp = standard(attrb)
	media = mean(attrb)
	nonoutliers = [False not in i for i in (attrb > (media - 2*desvp))]	# mantém exemplos dentro de 2 vezes o desvio pra baixo
	attrb = attrb[nonoutliers]
	clssb = clssb[nonoutliers]
	nonoutliers = [False not in i for i in (attrb < (media + 2*desvp))]	# mantém exemplos dentro de 2 vezes o desvio para cima
	attrb = attrb[nonoutliers]
	clssb = clssb[nonoutliers]

	# encestamento para remover ruídos
	bins(attrb)

	# classificação
	nn(clssb,attrb,ntreino)
	rochio(clssb,attrb,ntreino)




def exe5(folder='bases/'):
	print('Exercício 5')
	data = []
	with open(folder + 'Runner_num.txt') as f:
		for i in f:
			data.append(list(map(float, i.split())))

	data = np.array(data)

	# A
	print('\nLetra A')
	w = getw(data[:,0],data[:,1],1)
	print('w0:',w[0])
	print('w1:',w[1])

	# B
	print('\nLetra B')
	print('Para 2020:', np.dot(w,[1,2020]))

	# C
	print('\nLetra C')
	kendall(data[:,0],data[:,1],0.05)
	kendall(data[:,0],data[:,1],0.01)

	# D
	print('\nLetra D')
	pearson(data[:,0],data[:,1],0.05)
	pearson(data[:,0],data[:,1],0.01)




def exe6(folder='bases/'):
	print('\nExercício 6')
	data = []
	with open(folder + 'auto-mpg.data.txt') as f:
		for i in f:
			l = i.split()
			if('?' not in l):
				data.append(list(map(float,l[:8])))

	data = np.array(data)

	# A
	print('\nLetra A')
	# A última coluna pode ser removida por se tratar do nome do carro, desnecessário para a análise
	ntreino = 150
	# print(data)
	w = getw(data[:ntreino,1:], data[:ntreino,0])
	teste = np.hstack((np.array([[1] * (data.shape[0] - ntreino)]).T, data[ntreino:,1:]))
	f = (np.dot(teste,w) - data[ntreino:,0]) ** 2
	L = f.sum() / (data.shape[0] - ntreino)
	print('Modelo:',w)
	print('RMSE:',L)

	# B
	print('\nLetra B')
	w = snedecor(x = data[ntreino:,1:], y = data[ntreino:,0], w = w)
	f = (np.dot(teste,w) - data[ntreino:,0]) ** 2
	L = f.sum() / (data.shape[0] - ntreino)
	print('Novo modelo:',w)
	print('RMSE:',L)




def exe7(folder='bases/'):
	print('\nExercício 7')
	data = []
	with open(folder + 'auto-mpg.data.txt') as f:
		for i in f:
			l = i.split()
			if('?' not in l):
				data.append(list(map(float,l[:8])))

	data = np.array(data)

	# A
	print('\nLetra A')

def main():
	# exe1()
	# exe2()
	# exe3()
	# exe5()
	# exe6()
	exe7()

if(__name__ == '__main__'):
	main()
