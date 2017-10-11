import numpy as np
import math

N = 5
POS_AMP = 2
NEG_AMP = .5
POS_VAR = .5
NEG_VAR = 2

def get_initial():
	return np.random.rand(N,N)

def get_coords(flat=False):
	coords = np.array([[(i,j) for j in range(N)] for i in range(N)])
	if numpify:
		return coords
	return coords.reshape((N*N,2))

def gauss_map(center, amp, var):
	coords = get_coords(flat=False)
	centers = np.hstack(center*(N*N)).reshape((N,N,2)).astype(np.float32)
	diff = np.square(coords - centers)
	dist_squared = np.sum(diff, axis=2)
	gauss = amp * np.power(math.e, -dist_squared/var)
	return gauss

def weighted_maps(weight, pos, neg):
	pass

def time_step(current):
	coords = get_coords(flat=True)
	total = np.zeros((N,N))
	for c in coords:
		weight = current[c]
		posMap = gaussMap(c, POS_AMP, POS_VAR)
		negMap = gaussMap(c, NEG_AMP, NEG_VAR)
		net = weighted_maps(weight, posMap, negMap)
		total += net
	return current + total



print gauss_map((3,4), 1, 1)