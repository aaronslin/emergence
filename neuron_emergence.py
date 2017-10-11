import numpy as np
import math

N = 20
POS_AMP = 2
NEG_AMP = .5
POS_VAR = .5
NEG_VAR = 2

def sigmoid(x):
	sigmoid = 1 / (1 + np.exp(-x))
	shifted = (sigmoid-0.5)*2
	return shifted

def get_initial():
	return np.random.uniform(low=-1.0, high=1.0, size=(N,N))

def get_coords(flat=False):
	coords = [(i,j) for j in range(N) for i in range(N)]
	if flat:
		return coords
	return np.array(coords).reshape((N,N,2))

def gauss_map(center, amp, var):
	coords = get_coords(flat=False)
	centers = np.hstack(center*(N*N)).reshape((N,N,2)).astype(np.float32)
	diff = np.square(coords - centers)
	dist_squared = np.sum(diff, axis=2)
	gauss = amp * np.power(math.e, -dist_squared/var)
	# Unclear if you actually want to be normalizing at this step
	normal = gauss/np.sum(gauss)
	return normal

def weighted_maps(weight, pos, neg):
	# Assumes that weight is a sigmoid in (-1, 1)
	return (weight+1)*pos + (weight-1)*neg

def time_step(current):
	coords = get_coords(flat=True)
	totalChange = np.zeros((N,N))
	for c in coords:
		weight = current[c]
		posMap = gauss_map(c, POS_AMP, POS_VAR)
		negMap = gauss_map(c, NEG_AMP, NEG_VAR)
		net = weighted_maps(weight, posMap, negMap)
		totalChange += net
	next = current + totalChange
	# Unclear if you actually want to be sigmoiding here
	next = sigmoid(next)
	return next

a = get_initial()
for i in range(12):
	print a
	a = time_step(a)




