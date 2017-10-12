import numpy as np
import math
import matplotlib.pyplot as plt

N = 20
AMP_MULTIPLIER = .1
POS_AMP = 2 * AMP_MULTIPLIER
NEG_AMP = -1 * AMP_MULTIPLIER
POS_VAR = N * .1
NEG_VAR = N * .2

DECAY_RATE = 0.8

def sigmoid(x):
	sigmoid = 1 / (1 + np.exp(-x))
	return sigmoid

def shifted_sigmoid(x):
	return (sigmoid(x)-0.5)*2

def get_initial():
	return np.random.uniform(low=0.0, high=1.0, size=(N,N))

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
	# Nope, you don't, because it gives the posmap a huge advantage
	return gauss

def weighted_maps(weight, pos, neg):
	# Assumes that weight is a sigmoid in (-1, 1)
	return (weight+1)*pos + (weight-1)*neg

def on_off_center_influence(weight, center):
	# Assumes that weight is a sigmoid in (0, 1)
	posMap = gauss_map(center, POS_AMP, POS_VAR)
	negMap = gauss_map(center, NEG_AMP, NEG_VAR)
	return weight * (posMap + negMap)

def time_step(current):
	coords = get_coords(flat=True)
	totalChange = np.zeros((N,N))
	for center in coords:
		weight = current[center]
		net = on_off_center_influence(weight, center)
		totalChange += net
	next = totalChange
	# Unclear if you actually want to be sigmoiding here
	# Probably not, because each cell acts indepedently

	#next = sigmoid(next)
	return next

def EI_update(w_E, w_I, center):
	# Assumes that weight is a sigmoid in (0, 1)
	posMap = gauss_map(center, POS_AMP, POS_VAR)
	negMap = gauss_map(center, NEG_AMP, NEG_VAR)
	net_E = posMap * w_E + negMap * w_I

	net_I = np.zeros((N,N))
	net_I[center] = w_E + w_I * DECAY_RATE
	return None


def step_EI_network(excite, inhibit):
	coords = get_coords(flat=True)
	ex_change = np.ones((N,N))
	in_change = np.ones((N,N))
	for center in coords:
		w_E = excite[center]
		w_I = inhibit[center]
		net_E, new_I = EI_update(w_E, w_I, center)
		# Incomplete, because we might not need another layer


def display_heatmap(img, index=0, prefix="heatmap"):
	plt.imshow(a, cmap='hot', interpolation='nearest')
	#plt.show()
	plt.savefig(prefix+str(index)+".png")

NUM_ITERS = 20
a = get_initial()
for i in range(NUM_ITERS):
	print "Step:",i
	print a
	display_heatmap(a, i, prefix="locality")
	a = time_step(a)

"""

We can try encoding the probability that each cell fires another cell.
But I think there's a chance that all of the cells will collect at 0 or 1.

We'll have to do hyperparameter search for sure.

X X X
X Y X <-- if all of these on, how do you prevent Y from staying on?
X X X

We can also try influencing on a fixed number of cells around
Where does life come from once it dies out in a region?


"""

