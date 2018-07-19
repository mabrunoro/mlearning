#!/usr/bin/env python3

import numpy as np
import scipy.stats as stats
import random
import math
import sys
import matplotlib.pyplot as plt

colors = ['r', 'b', 'g', 'c', 'm', 'y']
shapes = ['o', '^', '*', 's', 'p', 'v']

def debug(arg):
	print(arg)
	sys.exit(0)

def euclidian(arg1,arg2):
	s = 0
	for i in range(len(arg1) - 1):	# last element is class number - should be ignored
		s += (arg1[i] - arg2[i])*(arg1[i] - arg2[i])
	return math.sqrt(s)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def sigmoidder(x):
	return x * (1-x)


class Perceptron:
	def __init__(self, nin=2, n=0.25, act=(lambda x: np.heaviside(x, 0.5)), ran=False):
		if(ran):
			self.w = np.random.random(nin+1)*2 - 1
		else:
			self.w = np.zeros(nin+1)	# pesos + bias
		self.phi = act	# função de ativação
		self.n = n	# taxa aprendizado

	def train(self, x, c):
		errt = 0
		for i in range(x.shape[0]):
			y = self.phi(np.sum(x[i] * self.w))
			err = c[i] - y
			errt += abs(err)
			self.w = self.w + self.n * x[i] * err
		# print(self.w)

	def fun(self, x):
		return self.phi(np.sum(self.w * x))

	def line(self, x):	# only works for 2D plots
		# return lambda x: (self.w[0] + self.w[1]*x)/(-self.w[2])
		return (self.w[0] + self.w[1]*x)/(-self.w[2])

class NN:
	def __init__(self, nin=4, nhid=16, nout=3, n=0.25, elm=False):
		self.nin = nin
		self.nhid = nhid
		self.nout = nout
		self.n = n
		self.build(elm=elm)

	def build(self, elm=False):
		self.hidden = np.array([ Perceptron(nin=self.nin, act=sigmoid, ran=elm) for i in range(self.nhid) ])
		self.output = np.array([ Perceptron(nin=self.nhid, act=sigmoid) for i in range(self.nout) ])

	def fun(self, X):
		hidout = np.array([1.0] + [ p.fun(X) for p in self.hidden ])
		outout = np.array([ p.fun(hidout) for p in self.output ])
		return (hidout,outout)

	def train(self, X, Y, errt = 0):
		for i in range(X.shape[0]):
			h,y = self.fun(np.array(X[i]))
			errt += (np.sum(Y[i] - y)/2)**2
			delm = (y - Y[i]) * y * (1 - y)
			deli = np.zeros(self.nhid + 1)

			for j in range(self.nout):
				deli += delm[j] * self.output[j].w

			for j in range(self.nout):
				for k in range(self.nhid + 1):
					self.output[j].w[k] = self.output[j].w[k] - self.n * delm[j] * h[k]

			for j in range(self.nhid):
				for k in range(self.nin + 1):
					# print(j,k,X[i])
					self.hidden[j].w[k] = self.hidden[j].w[k] - self.n * h[j] * (1 - h[j]) * deli[j] * X[i][k]
		errt /= X.shape[0]
		return errt

	def elm(self, X, Y):
		H = []
		for i in range(X.shape[0]):
			H.append(self.fun(X[i])[0])
		H = np.array(H)
		if(H.shape[0] > H.shape[1]):
			piH = np.matmul(np.power(np.matmul(H.T, H), -1), H.T)
		else:
			piH = np.matmul(H.T, np.power(np.matmul(H, H.T), -1))
		B = np.matmul(piH, Y)
		self.output[0].w = B


def exe01(folder,lfiles):
	print("\nExercise 1")
	for file in lfiles:
		data = []
		with open(folder+file[0]) as f:
			for i in f:
				data.append(list(map(float,i.split())))
		data = np.array(data)
		nclusters = file[1]	# number of clusters

		print("File:", file[0])
		print("Number of clusters:", nclusters)
		clusters = []
		samples = random.sample(range(data.shape[0]), data.shape[0])

		# K-Means
		print("K-Means")
		for i in range(nclusters):	# initiates clusters with random samples
			clusters.append([data[samples[i]]])

		for i in samples[nclusters:]:	# takes the remaining samples and cluster them
			mapa = list(map(lambda x : euclidian(data[i],x), [np.mean(j, axis=0) for j in clusters]))
			best = np.argmin(mapa)
			# print(best,mapa,data[i],[np.mean(j, axis=0) for j in clusters])
			clusters[best].append(data[i])

		for i in range(len(clusters)):
			k = np.array(clusters[i])
			u = np.unique(k[:,2], return_counts=True)
			c = np.argmax(u[1])
			# print(u)
			print("Cluster ", i, " class: ", u[0][c], ". Accuracy: ", u[1][c]/len(clusters[i]))

		for i in range(len(clusters)):
			for j in clusters[i]:
				plt.scatter(j[0], j[1], c=colors[i], marker=shapes[int(j[2])])
		plt.title(file[0])
		plt.show()

		# Hierarchical Clustering
		print("Hierarchical Clustering")
		clusters = [np.array([data[i]]) for i in range(data.shape[0])]
		while(len(clusters) > nclusters):
			best = None
			for i in range(len(clusters)-1):
				for j in range(i+1,len(clusters)):
					if(best is None):
						best = (i, j, euclidian(clusters[i][0],clusters[j][0]))
					else:
						d = euclidian(clusters[i].mean(axis=0), clusters[j].mean(axis=0))
						if(d < best[2]):
							best = (i, j, d)
			clusters[best[0]] = np.vstack((clusters[best[0]], clusters[best[1]]))
			del clusters[best[1]]

		for i in range(len(clusters)):
			k = clusters[i]
			u = np.unique(k[:,2], return_counts=True)
			c = np.argmax(u[1])
			# print(u)
			print("Cluster ", i, " class: ", u[0][c], ". Accuracy: ", u[1][c]/len(clusters[i]))

		for i in range(len(clusters)):
			for j in clusters[i]:
				plt.scatter(j[0], j[1], c=colors[i], marker=shapes[int(j[2])])
		plt.title(file[0])
		plt.show()


def exe02():
	print('\nExercise 2')
	print('On report.')


def exe03():
	print("\nExercise 3")
	X = np.array([[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]])
	Y = np.array([0, 1, 1, 1])
	p = Perceptron()
	for i in range(7):
		p.train(X,Y)

	print("Weights:", p.w)

	y = np.array([p.fun(i) for i in X])

	# plot 1 points of OR
	c = y == 1
	plt.scatter(X[c,1], X[c,2], c='g')

	# plot 0 points of OR
	c = y != 1
	plt.scatter(X[c,1], X[c,2], c='r')
	plt.plot([0,0.5],[p.line(0), p.line(0.5)],c='k')
	plt.show()


def exe04(file):
	print("\nExercise 4")
	data = []
	clasn = []
	clas = []

	with open(file) as f:
		for line in f:
			l = line.split(',')
			data.append([1.0] + list(map(float, l[:-1])))
			clasn.append(l[-1])
			if(l[-1] == 'Iris-setosa'):
				# clas.append([1,0,0])
				clas.append(1)
			elif(l[-1] == 'Iris-versicolor'):
				# clas.append([0,1,0])
				clas.append(2)
			else:
				# clas.append([0,0,1])
				clas.append(3)

	data = np.array(data)
	clas = np.array(clas)
	clasn = np.array(clasn)

	quarto = math.floor(data.shape[0] / 4)
	samples = random.sample(range(len(data)), len(data))

	# A
	print("\n\tA")
	acc = []
	best = (4,10)
	for t in range(4,16):
		m = NN(nhid=t,nout=1)
		m.train(data[samples[:2*quarto]],clas[samples[:2*quarto]])
		err = m.train(data[samples[2*quarto:3*quarto]],clas[samples[2*quarto:3*quarto]])
		if(err < best[1]):
			best = (t,err)

	m = NN(nhid=best[0],nout=1)
	for iter in range(4):
		samples = random.sample(range(len(data)), len(data))
		m.train(data[samples[:3*quarto]],clas[samples[:3*quarto]])
		_,o = m.fun(data[samples[3*quarto:]])
		v = np.sum(abs(o - clas[samples[3*quarto:]]))/(o.shape[0])
		acc.append(v)
	print("Accuracy:", acc)
	print("Mean:", np.mean(acc))
	print("Standard deviation:",np.std(acc))
	# print([i.w for i in m.output])

	# B
	print("\n\tB")
	acc = []
	best = (4,10)
	samples = random.sample(range(len(data)), len(data))
	for t in range(4,16):
		e = NN(nhid=t, nout=1, elm=True)
		e.elm(data[samples[:2*quarto]], clas[samples[:2*quarto]])
		_,o = e.fun(data[samples[2*quarto:3*quarto]])
		err = np.sum(abs(o - clas[samples[2*quarto:3*quarto]]))/(o.shape[0])
		if(err < best[1]):
			best = (t,err)

	e = NN(nhid=best[0], nout=1, elm=True)
	for iter in range(4):
		samples = random.sample(range(len(data)), len(data))
		e.elm(data[samples[:3*quarto]], clas[samples[:3*quarto]])
		_,o = e.fun(data[samples[3*quarto:]])
		v = np.sum(abs(o - clas[samples[3*quarto:]]))/(o.shape[0])
		acc.append(v)
	print("Accuracy:", acc)
	print("Mean:", np.mean(acc))
	print("Standard deviation:",np.std(acc))


def exe05(file):
	print("\nExercise 5")
	data = []
	with open(file) as f:
		for l in f:
			data.append(list(map(float,l.split())))
	data = np.array(data) / 100
	# print(data.shape)
	dif = data[:,0] - data[:,1]

	# Algoritmo para remover diferenças iguais deliberadamente removido do código.
	# Aparentemente, deveria ser realizada uma análise das diferenças iguais (em módulo) com
	# o sinal para que as médias não sejam desbalanceadas na remoção
	# sorted = np.argsort(abs(dif))
	# remove = []
	# ig = 0
	#
	# for i in range(sorted.shape[0] - 1):
	# 	print(dif[sorted[i]], dif[sorted[i+1]])
	# 	if(dif[sorted[i]] == dif[sorted[i+1]]):
	# 		ig += 1
	# 	else:
	# 		if((ig > 0) and (ig % 2 == 0)):
	# 			remove.append(i)
	# 		ig = 0
	# print(remove)

	ranks = stats.rankdata(abs(dif))
	Rp = np.sum(ranks[dif > 0])
	Rm = np.sum(ranks[dif < 0])
	print("R+:", Rp)
	print("R-:", Rm)


def main(folder='bases/'):
	# exe01(folder, [('spiral.txt',3), ('jain.txt',2)])
	# exe02()
	# exe03()
	exe04(folder + 'iris.data.txt')
	# exe05(folder + "wilcoxon.txt")

if __name__ == '__main__':
	main()
