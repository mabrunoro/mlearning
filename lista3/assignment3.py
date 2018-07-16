#!/usr/bin/env python3

import numpy as np
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


class Perceptron:
	def __init__(self, nin=2, n=0.25, act=lambda x: np.heaviside(x, 0.5)):
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


def exe01(folder,lfiles):
	print("\n\Exercise 1")
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
			print(u)
			print("Cluster ", i, " class: ", u[0][c], ". Accuracy: ", u[1][c]/len(clusters[i]))

		for i in range(len(clusters)):
			for j in clusters[i]:
				plt.scatter(j[0], j[1], c=colors[i], marker=shapes[int(j[2])])
		plt.title(file[0])
		plt.show()

		# Hierarchical Clustering
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
			print(u)
			print("Cluster ", i, " class: ", u[0][c], ". Accuracy: ", u[1][c]/len(clusters[i]))

		for i in range(len(clusters)):
			for j in clusters[i]:
				plt.scatter(j[0], j[1], c=colors[i], marker=shapes[int(j[2])])
		plt.title(file[0])
		plt.show()


def exe02(file):
	print('\nExercise 2')
	print('On report.')


def exe03():
	X = np.array([[1, 0, 0],[1, 0, 1],[1, 1, 0],[1, 1, 1]])
	Y = np.array([0, 1, 1, 1])
	p = Perceptron()
	for i in range(7):
		p.train(X,Y)

	y = np.array([p.fun(i) for i in X])

	# plot 1 points of OR
	c = y == 1
	plt.scatter(X[c,1], X[c,2], c='g')

	# plot 0 points of OR
	c = y != 1
	plt.scatter(X[c,1], X[c,2], c='r')
	plt.plot([0,0.5],[p.line(0), p.line(0.5)],c='k')
	plt.show()


def main(folder='bases/'):
	# exe01(folder, [('spiral.txt',3), ('jain.txt',2)])
	# exe02()
	# exe03()
	exe04(folder+'iris.data.txt')

if __name__ == '__main__':
	main()