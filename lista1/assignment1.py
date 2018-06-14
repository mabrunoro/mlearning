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

def nn(clss,attr,ntreino,ret=False):
	clssa = clss.copy()
	for i in range(ntreino,attr.shape[0]):	# para cada item de teste
		distancia = 0
		for j in range(ntreino):	# passa por cada item de treino
			if(j == 0):
				if(len(attr.shape) < 2):
					distancia = np.linalg.norm(attr[i] - attr[j])
				elif(ret):
					distancia = np.linalg.norm(attr[i,:] - attr[j,:])
				else:
					distancia = np.linalg.norm(attr[i,2:] - attr[j,2:])
				clssa[i] = clss[j]
			else:
				if(len(attr.shape) < 2):
					aux = np.linalg.norm(attr[i] - attr[j])
				elif(ret):
					aux = np.linalg.norm(attr[i,:] - attr[j,:])
				else:
					aux = np.linalg.norm(attr[i,2:] - attr[j,2:])
				if(aux < distancia):
					distancia = aux
					clssa[i] = clss[j]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)

	if(len(res[0]) < 2):
		if(res[0][0]):
			# print('Acurácia NN: 100')
			# return 100
			acc = 100
		else:
			# print('Acurácia NN: 0')
			# return 0
			acc = 0
	elif(res[0][0]):
		# print('Acurácia NN:', 100*res[1][0]/(clssa.shape[0]-ntreino))
		# return 100*res[1][0]/(clssa.shape[0]-ntreino)
		acc = 100*res[1][0]/(clssa.shape[0]-ntreino)
	else:
		# print('Acurácia NN:', 100*res[1][1]/(clssa.shape[0]-ntreino))
		# return 100*res[1][1]/(clssa.shape[0]-ntreino)
		acc = 100*res[1][1]/(clssa.shape[0]-ntreino)

	if(not ret):
		print('Acurácia NN:',acc)
	return acc

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
				clssa[i] = classes[j]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)

	if(len(res[0]) < 2):
		if(res[0][0]):
			print('Acurácia Rochio: 100')
		else:
			print('Acurácia Rochio: 0')
	elif(res[0][0]):
		print('Acurácia Rochio:', 100*res[1][0]/(clssa.shape[0]-ntreino))
	else:
		print('Acurácia Rochio:', 100*res[1][1]/(clssa.shape[0]-ntreino))

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

def rmse(x, y, w):
	t = np.dot(x,w)
	return math.sqrt(((y - t)**2).sum()/y.shape[0])

def mape(x, y, w):
	return (100/y.shape[0])*(abs((y - np.dot(x,w))/y)).sum()

def qme(x, y, w):
	N = x.shape[0]
	if(len(x) == 1):
		p = 1
	else:
		p = x.shape[1]
	return ((y - np.dot(x,w)) ** 2).sum()/(N-p)

def rsq(x, y, w):
	num = ((y - np.dot(x,w)) ** 2).sum()
	yb = y.mean()
	den = ((y - yb) ** 2).sum()
	return 1 - (num/den)

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
	amostras = random.sample(range(x.shape[0]), N)
	print('\nUsando',N,'amostras...')
	p = np.cov(x[amostras], y[amostras], rowvar=False)[0,1] / math.sqrt(variance(x[amostras]) * variance(y[amostras]))

	to = p*math.sqrt(N-1)/math.sqrt(1-p*p)
	t = st.t.ppf(sig/2,N-2)
	if(abs(to) > t):
		print("Pearson: hipótese nula rejeitada. Existe a possibilidade de haver dependência entre 'x' e 'y' com",1-sig,'de significância.','\nto =',to,'\nt =',t)
	else:
		print('Pearson: hipótese nula aceita para',1-sig,'de significância.\nto =',to,'\nt =',t)

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

def ransac(x, y, s, pot=1, eps=0.2, p=0.99, rmse = 50):
	T = x.shape[0] * (1 - eps)
	tau = rmse # math.sqrt(3.84 * (np.var(y) ** 2))
	L = math.ceil(math.log10(1-p)/math.log10(1-((1-eps)**s)))
	mk = [0,0]
	# print('Ransac\tT:',T,'tau:',tau,'L:',L)

	for i in range(L):
		samples = random.sample(range(x.shape[0]),4)
		w = getw(x[samples[:s],1:(1+pot)], y[samples[:s]])
		t = np.dot(x[:,:(1+pot)],w)
		dentro = abs(t - y) <= tau
		co = np.unique(dentro, return_counts=True)
		if(len(co[0]) < 2):
			continue
		if(co[0][0]):
			k = co[1][0]
		else:
			k = co[1][1]
		if(k >= T):
			w = getw(x[dentro,1:(1+pot)], y[dentro])
			return w
		elif(mk[0] < k):
				mk[1] = dentro
	return getw(x[mk[1],1:(1+pot)], y[mk[1]])

def rochiomah(clss, attr, ntreino):
	classes = np.unique(clss)
	clssa = clss.copy()
	medias = []
	for i in classes:
		medias.append(mean(attr[:ntreino][clss[:ntreino] == i]))
	eta = np.cov(attr[ntreino:],rowvar=False)
	for i in range(ntreino,clss.shape[0]):
		diff = attr[i] - medias[0]
		distancia = math.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(eta)),diff))
		clssa[i] = classes[0]
		for j in range(1,len(medias)):
			# aux = np.linalg.norm(attr[i] - medias[j])
			diff = attr[i] - medias[j]
			aux = math.sqrt(np.dot(np.dot(diff.T, np.linalg.inv(eta)),diff))
			if(aux < distancia):
				distancia = aux
				clssa[i] = classes[j]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)
	if(res[0][0]):
		# print('\nAcurácia Rochio Mahalanobis:', 100*res[1][0]/(clssa.shape[0]-ntreino))
		acc = 100*res[1][0]/(clssa.shape[0]-ntreino)
	else:
		# print('\nAcurácia Rochio Mahalanobis:', 100*res[1][1]/(clssa.shape[0]-ntreino))
		acc = 100*res[1][1]/(clssa.shape[0]-ntreino)

	aux = clss[ntreino:]	# obtém todos exemplos dados como positivos (TP + FP)
	aux = aux[clssa[ntreino:] == 1]

	res = np.unique(aux == 1, return_counts=True)	# descobre quantos exemplos dados como positivos realmente o são
	if(res[0][0]):
		tp = res[1][0]
		fp = res[1][1]
	else:
		tp = res[1][1]
		fp = res[1][0]
	P = 100*tp/(tp+fp)

	aux = clss[ntreino:][clssa[ntreino:] == 0]	# obtém todos os exemplos dados como negativos
	res = np.unique(aux == 1, return_counts=True)	# descobre quantos exemplos são falso negativos
	if(res[0][0]):
		fn = res[1][0]
	else:
		fn = res[1][1]
	R = 100*tp/(tp+fn)

	return (acc,P,R)

def knn(clss,attr, ntreino,k=5):
	clssa = clss.copy()
	for i in range(ntreino,clss.shape[0]):
		dists = np.linalg.norm(attr[i] - attr[:,:ntreino],axis=1)
		largedists = np.argpartition(dists,k)[:k]
		unq = np.unique(clss[largedists], return_counts=True)
		if(len(unq[1]) < 2):
			clssa[i] = unq[0][0]
		elif(unq[1][0] > unq[1][1]):
			clssa[i] = unq[0][0]
		else:
			clssa[i] = unq[0][1]

	res = np.unique(clss[ntreino:] == clssa[ntreino:], return_counts=True)
	if(res[0][0]):
		# print('\nAcurácia Rochio Mahalanobis:', 100*res[1][0]/(clssa.shape[0]-ntreino))
		acc = 100*res[1][0]/(clssa.shape[0]-ntreino)
	else:
		# print('\nAcurácia Rochio Mahalanobis:', 100*res[1][1]/(clssa.shape[0]-ntreino))
		acc = 100*res[1][1]/(clssa.shape[0]-ntreino)

	aux = clss[ntreino:][clssa[ntreino:] == 1]	# obtém todos exemplos dados como positivos (TP + FP)
	res = np.unique(aux == 1, return_counts=True)	# descobre quantos exemplos dados como positivos realmente o são
	if(len(res[0]) == 0):
		tp = 0
		fp = 0
	elif(len(res[0]) == 1):
		if(res[0][0]):
			tp = res[1][0]
			fp = 0
		else:
			tp = 0
			fp = res[1][0]
	elif(res[0][0]):
		tp = res[1][0]
		fp = res[1][1]
	else:
		tp = res[1][1]
		fp = res[1][0]
	if(tp+fp == 0):
		P = 0
	else:
		P = 100*tp/(tp+fp)

	aux = clss[ntreino:][clssa[ntreino:] == 0]	# obtém todos os exemplos dados como negativos
	res = np.unique(aux == 1, return_counts=True)	# descobre quantos exemplos são falso negativos

	if(res[0][0]):
		fn = res[1][0]
	else:
		if(len(res[0]) < 2):
			fn = 0
		else:
			fn = res[1][1]
	if(tp+fn == 0):
		R = 0
	else:
		R = 100*tp/(tp+fn)

	return (acc,P,R)

def sfs(clss, attr, aval, ntreino, lim=5):
	if((len(attr.shape) > 1) and (lim >= attr.shape[1])):
		return (list(range(attr.shape[1])),0)

	best = [0, [0]]
	l = list(range(attr.shape[1]))
	for j in range(attr.shape[1]):
		if(j == 0):
			best[0] = aval(clss, attr[:,j], ntreino, ret=True)
			best[1][0] = 0
		else:
			aux = aval(clss, attr[:,j], ntreino, ret=True)
			if(aux > best[0]):
				best[0] = aux
				best[1][0] = j

	l.remove(best[1][0])
	laux = [best[1][0]]

	for i in range(1,lim):
		best[1].append(-1)
		best[0] = 0
		laux.append(0)
		for j in l:
			laux[i] = j
			aux = aval(clss, attr[:,laux], ntreino, ret=True)
			if(aux > best[0]):
				best[0] = aux
				best[1][i] = j
			# print(best[1], j, aux, best[0])
		# if((best[0] >= 99) or (best[1][i] == -1)):
		# 	best[1] = best[1][:-1]
		# 	break
		l.remove(best[1][i])
	return (np.array(best[1]),best[0])

def sbe(clss, attr, aval, ntreino, lim=5):
	if((len(attr.shape) > 1) and (lim >= attr.shape[1])):
		return (list(range(attr.shape[1])),0)

	best = [0, np.array(range(attr.shape[1]))]
	l = best[1]

	for j in range(attr.shape[1]):
		if(j == 0):
			best[0] = aval(clss, np.delete(attr, j, axis=1), ntreino, ret=True)
			l = np.delete(best[1], j)
		else:
			aux = aval(clss, np.delete(attr, j, axis=1), ntreino, ret=True)
			if(aux > best[0]):
				best[0] = aux
				l = np.delete(best[1], j)

	best[1] = l

	while(best[1].shape[0] > lim):
		best[0] = 0

		for j in range(best[1].shape[0]):
			laux = np.delete(best[1], j)
			aux = aval(clss, attr[:,laux], ntreino, ret=True)
			if(aux > best[0]):
				best[0] = aux
				l = laux
			# print(best[1], j, aux, best[0])
		# if((best[0] >= 99) or (best[1][i] == -1)):
		# 	best[1] = best[1][:-1]
		# 	break
		best[1] = l

	return (best[1],best[0])



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
	ntreino = math.floor(ntreino*attrb.shape[0]/attr.shape[0])
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
	# f = (np.dot(teste,w) - data[ntreino:,0]) ** 2
	# L = f.sum() / (data.shape[0] - ntreino)
	print('Modelo:',w)
	print('RMSE:',rmse(teste, data[ntreino:,0],w))

	# B
	print('\nLetra B')
	w = snedecor(x = data[ntreino:,1:], y = data[ntreino:,0], w = w)
	# f = (np.dot(teste,w) - data[ntreino:,0]) ** 2
	# L = f.sum() / (data.shape[0] - ntreino)
	print('Novo modelo:',w)
	print('RMSE:',rmse(teste,data[ntreino:,0],w))




def exe7(folder='bases/'):
	print('\nExercício 7')
	data = []
	with open(folder + 'Polinômio.txt') as f:
		for i in f:
			l = i.split()
			if('?' not in l):
				data.append(list(map(float,l)))

	data = np.array(data)
	# print(data)

	# A
	print('\nLetra A')
	samples = random.sample(range(data.shape[0]),data.shape[0])
	ntreino = math.floor(0.7 * data.shape[0])
	xtreino = np.vstack((np.ones(ntreino), data[samples[:ntreino],0]))
	ytreino = data[samples[:ntreino],1]
	xteste = np.vstack((np.ones(len(samples) - ntreino), data[samples[ntreino:],0]))
	yteste = data[samples[ntreino:],1]

	wlinear = getw(xtreino.T[:,1], ytreino, 1)
	print('Modelo:',wlinear)
	print('RMSE treino:', rmse(xtreino.T,ytreino,wlinear))
	print('MAPE treino:', mape(xtreino.T,ytreino,wlinear))
	print('RMSE teste:', rmse(xteste.T,yteste,wlinear))
	print('MAPE teste:', mape(xteste.T,yteste,wlinear))

	plt.scatter(data[:,0], data[:,1], c='r', edgecolors='k', label='dados')
	plt.scatter(xtreino.T[:,1], np.dot(xtreino.T,wlinear), c='b', edgecolors='k', label='treino')
	plt.scatter(xteste.T[:,1], np.dot(xteste.T,wlinear), c='g', edgecolors='k', label='teste')
	plt.legend()
	plt.show()
	# O modelo não se ajustou bem aos dados, pois os dados possuem características polinomiais

	# B
	print('\nLetra B')
	qan = qme(xtreino.T, ytreino, wlinear)
	ran = rsq(xtreino.T, ytreino, wlinear)
	print('QME modelo linear:',qan)
	print('R^2 modelo linear:',ran)

	qat = qan
	rat = ran

	pot = 1

	while((qat <= qan) or (rat >= ran)):
		qan = qat
		ran = rat
		pot += 1
		xtreino = np.vstack((xtreino, xtreino[1,:] ** pot))
		xteste = np.vstack((xteste, xteste[1,:] ** pot))
		wpol = getw(xtreino.T[:,1:], ytreino, pot)
		qat = qme(xtreino.T, ytreino, wpol)
		rat = rsq(xtreino.T, ytreino, wpol)
	print('Grau do melhor modelo:',pot-1)
	print('QME modelo polinomial:',qan)
	print('R^2 modelo polinomial:',ran)

	print('Melhor modelo polinomial:',wpol)
	print('RMSE treino:', rmse(xtreino.T,ytreino,wpol))
	print('MAPE treino:', mape(xtreino.T,ytreino,wpol))
	print('RMSE teste:', rmse(xteste.T,yteste,wpol))
	print('MAPE teste:', mape(xteste.T,yteste,wpol))

	plt.scatter(data[:,0], data[:,1], c='r', edgecolors='k', label='dados')
	plt.scatter(xtreino.T[:,1], np.dot(xtreino.T,wpol), c='b', edgecolors='k', label='treino')
	plt.scatter(xteste.T[:,1], np.dot(xteste.T,wpol), c='g', edgecolors='k', label='teste')
	plt.legend()
	plt.show()
	# Não. A presença de outliers dificulta a obtenção de um bom modelo polinomial.

	# C
	print('\nLetra C')
	oldrmse = rmse(xtreino.T,ytreino,wpol)
	xtreino = xtreino[:2,:]
	xteste = xteste[:2,:]
	best = [np.array([]), oldrmse, 1]

	for i in range(2,pot):
		xtreino = np.vstack((xtreino, xtreino[1,:] ** i))
		xteste = np.vstack((xteste, xteste[1,:] ** i))
		wpol = ransac(xtreino.T, ytreino, s=20, pot=i, p=0.99, eps=0.2, rmse=oldrmse)
		# print(xtreino.shape,wpol.shape,i)
		nrmse = rmse(xtreino.T[:,:(pot+1)],ytreino,wpol)
		print('RMSE modelo Ransac de grau ',i,': ',nrmse,sep='')
		if(nrmse <= best[1] or (best[0].shape[0] == 0)):
			best[2] = i
			best[1] = nrmse
			best[0] = wpol

	print('Modelo polinomial Ransac:',best[0])
	print('RMSE treino:', rmse(xtreino.T[:,:(best[2]+1)],ytreino,best[0]))
	print('MAPE treino:', mape(xtreino.T[:,:(best[2]+1)],ytreino,best[0]))
	print('RMSE teste:', rmse(xteste.T[:,:(best[2]+1)],yteste,best[0]))
	print('MAPE teste:', mape(xteste.T[:,:(best[2]+1)],yteste,best[0]))

	plt.scatter(data[:,0], data[:,1], c='r', edgecolors='k', label='dados')
	plt.scatter(xtreino.T[:,1], np.dot(xtreino.T[:,:(best[2]+1)],best[0]), c='b', edgecolors='k', label='treino')
	plt.scatter(xteste.T[:,1], np.dot(xteste.T[:,:(best[2]+1)],best[0]), c='g', edgecolors='k', label='teste')
	plt.legend()
	plt.show()




def exe10(folder='bases/'):
	print('\nExercício 10')
	data = []
	with open(folder + 'HTRU2/HTRU_2.csv') as f:
		for i in f:
			l = i.split(',')
			if('?' not in l):
				data.append(list(map(float,l)))

	data = np.array(data)
	# print(data)

	ntreino = 6000
	niter = 5

	# A
	print('\nLetra A')
	acc = 0
	pre = 0
	rec = 0
	for i in range(niter):
		(a,p,r) = rochiomah(clss=data[:,-1],attr=data[:,:-1],ntreino=ntreino)
		acc += a
		pre += p
		rec += r
	acc /= 5
	pre /= 5
	rec /= 5
	print('Acurácia:',acc)
	print('Precisão:',pre)
	print('Revocação:',rec)

	# B
	print('\nLetra B')
	k = 1
	acc = 0
	while(acc < 95):
		k += 2
		acc,_,_ = knn(clss=data[:ntreino,-1],attr=data[:ntreino,:-1],ntreino=math.floor(ntreino/2),k=k)
		print('Acurácia para k = ',k,': ',acc,sep='')

	acc = 0
	pre = 0
	rec = 0
	for i in range(niter):
		(a,p,r) = knn(clss=data[:,-1],attr=data[:,:-1],ntreino=ntreino)
		acc += a
		pre += p
		rec += r

	acc /= 5
	pre /= 5
	rec /= 5
	print('Acurácia:',acc)
	print('Precisão:',pre)
	print('Revocação:',rec)

	# C
	print('\nLetra C')
	mask = np.ones(ntreino, dtype=bool)
	j = 0
	for i in range(ntreino):
		mask[i] = False
		acc,_,_ = knn(clss=data[:ntreino,-1][mask], attr=data[:ntreino,:-1][mask], ntreino=math.floor((ntreino-j-1)/2), k=3)

		if(acc >= 95):
			j += 1
		else:
			mask[i] = True
	acc = 0
	pre = 0
	rec = 0
	mask = np.hstack((mask,np.ones(data.shape[0]-ntreino, dtype=bool)))
	for i in range(niter):
		(a,p,r) = knn(clss=data[:,-1][mask],attr=data[:,:-1][mask],ntreino=ntreino)
		acc += a
		pre += p
		rec += r

	acc /= 5
	pre /= 5
	rec /= 5
	print('Acurácia:',acc)
	print('Precisão:',pre)
	print('Revocação:',rec)




def exe11(folder='bases/'):
	print('\nExercício 11')
	data = []
	with open(folder + 'wine.data.txt') as f:
		for i in f:
			l = i.split(',')
			if('?' not in l):
				data.append(list(map(float,l)))

	data = np.array(data)
	# print(data)

	# A
	print('\nLetra A')
	N = data.shape[0]
	samples = random.sample(range(N),N)
	data1 = data[samples[:math.floor(N/3)]]
	data2 = data[samples[math.floor(N/3):math.floor(2*N/3)]]
	data3 = data[samples[math.floor(2*N/3):]]

	ntreino = data1.shape[0]
	datasfs = np.vstack((data1,data2))

	atributos,_ = sfs(clss=datasfs[:,0], attr=datasfs[:,1:], aval=nn, ntreino=ntreino)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=(data1.shape[0]+data2.shape[0]))

	atributos,_ = sbe(clss=datasfs[:,0], attr=datasfs[:,1:], aval=nn, ntreino=ntreino)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=(data1.shape[0]+data2.shape[0]))

	# B
	print('\nLetra B')
	atributos,_ = sfs(clss=datasfs[:,0], attr=datasfs[:,1:], aval=nn, ntreino=ntreino, lim=10)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=(data1.shape[0]+data2.shape[0]))

	atributos,_ = sbe(clss=datasfs[:,0], attr=datasfs[:,1:], aval=nn, ntreino=ntreino, lim=10)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=(data1.shape[0]+data2.shape[0]))

	# C
	print('\nLetra C')
	# para simular o caso em que o mesmo conjunto de dados seja utilizado para treinamento e validação, é criado uma matriz duplicando os dados
	datac = np.vstack((datasfs,datasfs))
	ntreino = datasfs.shape[0]
	atributos,_ = sfs(clss=datac[:,0], attr=datac[:,1:], aval=nn, ntreino=ntreino)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=ntreino)

	atributos,_ = sbe(clss=datac[:,0], attr=datac[:,1:], aval=nn, ntreino=ntreino)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=ntreino)

	atributos,_ = sfs(clss=datac[:,0], attr=datac[:,1:], aval=nn, ntreino=ntreino, lim=10)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=ntreino)

	atributos,_ = sbe(clss=datac[:,0], attr=datac[:,1:], aval=nn, ntreino=ntreino, lim=10)
	print('Atributos selecionados:',atributos)
	nn(data[samples,0],data[samples][:,atributos],ntreino=ntreino)

def main():
	# exe1()
	# exe2()
	# exe3()
	# exe5()
	# exe6()
	# exe7()
	# exe10()
	# exe11()
	print('A resolução da lista foi implementada em funções.')
	print('As funções, que podem ser acessadas importando este arquivo, são:')
	print('\texe1()')
	print('\texe2()')
	print('\texe3()')
	print('\texe5()')
	print('\texe6()')
	print('\texe7()')
	print('\texe10()')
	print('\texe11()')

if(__name__ == '__main__'):
	main()
