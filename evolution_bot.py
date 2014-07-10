#!/usr/bin/python
import os
import re
import random
import numpy as np
from   math  import exp
from   rgkit import rg

def rand_plus_or_minus():
	return (random.random() * 2.0) - 1.0

class Perceptron:

	def __init__(self, layer_sizes=[], mats=None, random=False, horizontal=False):
		self.__weight_matrices = mats or []
		self.__sizes = layer_sizes
		self.__horizontal = horizontal
		self.__reset_states()
		if not mats:
			for i in range(len(layer_sizes) - 1):
				fromsize = layer_sizes[i]
				tosize = layer_sizes[i+1]
				mat = np.matrix(np.zeros((tosize, fromsize)))
				self.__weight_matrices.append(mat)
		if random:
			self.mutate(edit=1.0)

	def __str__(self):
		return "\n".join([str(mat) for mat in self.__weight_matrices])

	def __reset_states(self):
		self.__states = map(lambda s: np.zeros((s,1)), self.__sizes)

	@staticmethod
	@np.vectorize
	def sigmoid(x):
		return 1.0 / (1.0 + exp(-x))

	def copy(self):
		c = Perceptron()
		for mat in self.__weight_matrices:
			c.__weight_matrices.append(mat.copy())
		return c

	def mutate(self, edit=0.001):
		for mat in self.__weight_matrices:
			changes = np.matrix([edit*rand_plus_or_minus() for _ in xrange(mat.size)])
			changes = changes.reshape(mat.shape)
			mat += changes
		return self

	def set_input(self, which, value):
		assert -1 <= value <= 1
		assert 0 <= which < self.__sizes[0]
		self.__states[0][which] = value

	def set_inputs(self, input_vector):
		self.__states[0] = input_vector

	def propagate(self, clear=True, iterations=1):
		if not self.__horizontal: iterations = 1
		if clear:
			inpts = self.__states[0]
			self.__reset_states()
			self.__states[0] = inpts
		for i in xrange(iterations):
			for l in xrange(len(self.__sizes)-1):
				# print self.__states
				# print self.__weight_matrices
				# raw_input()
				self.__states[l+1] = Perceptron.sigmoid(self.__weight_matrices[l] * self.__states[l])

	def get_output(self, which):
		assert 0 <= which < self.__sizes[-1]
		return self.__sizes[which]

	def choose_output(self, strict=False):
		if strict:
			maxi = 0
			maxv = self.__states[-1][0]
			for i in xrange(1, self.__sizes[-1]):
				if self.__states[-1][i] > maxv:
					maxv = self.__states[-1][i]
					maxi = i
			return maxi, maxv
		else:
			summed = self.__states[-1].cumsum(1)
			index = random.random() * summed[-1]
			for i in xrange(0, self.__sizes[-1]):
				if index <= summed[i]:
					return i, self.__states[-1][i]
			return i, self.__states[-1][i]

	@staticmethod
	def mate(p0, p1):
		assert p0.__sizes == p1.__sizes
		assert p0.__horizontal == p1.__horizontal
		newp = Perceptron(layer_sizes=p0.__sizes, horizontal=p0.__horizontal)
		for l in xrange(len(newp.__sizes)-1):
			rows, cols = newp.__weight_matrices[l].shape
			for r in xrange(rows):
				for c in xrange(cols):
					newp.__weight_matrices[l][(r,c)] = random.choice([p0.__weight_matrices[l][(r,c)], p1.__weight_matrices[l][(r,c)]])
		return newp

	@staticmethod
	def load(file_path):
		p = None
		with open(file_path, "r") as f:
			sizes = [int(n) for n in f.readline().split()]
			p = Perceptron(sizes, random=False)
			mat_text = f.read()
			# parse numpy output back into matrices
			mat_text = re.sub(r"\[\[", "", mat_text)
			mat_text = re.sub(r"\]\s+\[", ";\n", mat_text)
			matstrings = mat_text.split("]]")
			p.__weight_matrices = [np.matrix(s) for s in matstrings]
		return p

	def save(self, directory, suffix=""):
		fname = "perceptron_" + ("_".join([str(l) for l in self.__sizes])) + suffix + ".dat"
		path = os.path.join(directory, fname)
		with open(path, "w") as f:
			f.write(" ".join([str(l) for l in self.__sizes]) + "\n")
			f.write(str(self))
		return path

def action_is_valid(act):
	if len(act) > 1 and ("invalid" in rg.loc_types(act[1]) or "obstacle" in rg.loc_types(act[1])):
		return False
	return True

class Robot:

	width = 3
	halfwidth = width / 2
	inputs = (width ** 2) * 4 # 4 = {ally hp, enemy hp, spawn, obstacle}
	outputs = 10 # 4 attack, 4 move, suicide, guard
	layers = [width**2]

	def global_init(self):
		if not hasattr(self, 'brain'):
			self.brain = Perceptron([Robot.inputs] + Robot.layers + [Robot.outputs])

	def set_brain(self, brain):
		self.brain = brain

	def construct_features(self, game):
		xc, yc = self.location
		features = np.zeros((Robot.inputs,1))
		i = 0
		for x in xrange(xc - Robot.halfwidth, xc + Robot.halfwidth + 1):
			for y in xrange(yc - Robot.halfwidth, yc + Robot.halfwidth + 1):
				bot = game.robots.get((x,y))
				if bot == None:
					features[(i, 0)] = 0.0
					features[(i+1, 0)] = 0.0
				else:
					features[(i, 0)]   = bot['hp'] / float(rg.settings.robot_hp) if bot['player_id'] == self.player_id else 0.0
					features[(i+1, 0)] = bot['hp'] / float(rg.settings.robot_hp) if bot['player_id'] != self.player_id else 0.0
				features[(i+2, 0)] = 1.0 if 'spawn' in rg.loc_types((x,y)) else 0.0
				features[(i+3, 0)] = 1.0 if 'obstacle' in rg.loc_types((x,y)) else 0.0
				i += 4
		return features

	def act(self, game):
		self.global_init()
		
		x,y = self.location

		features = self.construct_features(game)
		
		st = ""
		for yi in xrange(y-Robot.halfwidth, y+Robot.halfwidth+1):
			for xi in xrange(x-Robot.halfwidth, x+Robot.halfwidth+1):
				if 'invalid' in rg.loc_types((xi,yi)):
					st += " X "
				elif 'obstacle' in rg.loc_types((xi,yi)):
					st += " # "
				elif (xi,yi) in game.robots:
					if game.robots[(xi,yi)].player_id == self.player_id:
						st += " A "
					else:
						st += " E "
				else:
					st += "   "
			st += "\n"
		# print st
		# print features
		# raw_input()


		self.brain.set_inputs(features)
		self.brain.propagate()

		index, value = self.brain.choose_output(strict=False)

		choices = [
			['move', (x-1, y)],   ['move', (x+1, y)],   ['move', (x, y-1)],   ['move', (x, y+1)],
			['attack', (x-1, y)], ['attack', (x+1, y)], ['attack', (x, y-1)], ['attack', (x, y+1)],
			['guard'],
			['suicide']]
		blind_choice = choices[index]
		if not action_is_valid(blind_choice): return ['guard']
		return blind_choice

# tournament
if __name__ == '__main__':
	from rgkit.run import Options, Runner
	from rgkit.game import Player
	from optparse import OptionParser

	parser = OptionParser()
	parser.add_option("-s", "--seed",          type="int",   dest="seed",         default=0,    help="set seed for games' RNG")
	parser.add_option("-g", "--generations",   type="int",   dest="generations",  default=500,  help="number of generations")
	parser.add_option("-n", "--ngames",        type="int",   dest="ngames",       default=1,    help="number of games per generation")
	parser.add_option("-p", "--population",    type="int",   dest="population",   default=8,    help="population size")
	parser.add_option("-r", "--random",        type="int",   dest="random",       default=1,    help="number of random brains to add each generation")
	parser.add_option("-m", "--mutation-rate", type="float", dest="mutationrate", default=0.01, help="mutation rate")
	parser.add_option("-o", "--output-dir",    type="string",dest="savedir",      default="brains", help="directory to save brains")
	(options, leftover_args) = parser.parse_args()

	population = options.population
	newrandom = options.random
	ngames = options.ngames
	mutation_rate = options.mutationrate
	max_generation = options.generations
	rand_seed = options.seed
	save_dir = options.savedir
	save_every = 100

	random.seed(rand_seed)

	# load brains from any files left after opts have been parsed
	brains = []
	for arg in leftover_args:
		try:
			loaded = Perceptron.load(arg)
			if loaded is not None:
				brains.append(loaded)
		except: pass

	# fill rest of brain array with random brains
	layers = [Robot.inputs] + Robot.layers + [Robot.outputs]
	while len(brains) < population:
		brains .append(Perceptron(layers, random=True))
	scores = [0.0] * population

	def run_match(brains, id0, id1):
		r0 = Robot()
		r0.set_brain(brains[id0])
		p0 = Player(robot=r0)
		r1 = Robot()
		r1.set_brain(brains[id1])
		p1 = Player(robot=r1)

		opts = Options(n_of_games=ngames, game_seed=generation + rand_seed)
		r = Runner(players=[p0, p1], options=opts)
		results = r.run()
		swing = 0
		for res in results:
			swing += res[0] - res[1]
		return swing

	def choose_brain(brains, scores):
		# weighted choice based on score
		weights = [1.0 / (1 + exp(-s / (max(scores)+0.1))) for s in scores]
		idx = random.random() * sum(weights)
		s = 0.0
		for i in xrange(population):
			s += weights[i]
			if s >= idx:
				return brains[i]

	generation = 0
	while generation < max_generation:
		print "--- generation %d ---" % generation
		scores = [0.0] * population

		# run 1v1 matches
		for bot0 in xrange(population):
			for bot1 in xrange(bot0+1, population):
				# run 0 vs 1
				swing = run_match(brains, bot0, bot1)
				# run 1 vs 0 (left/right symmetry)
				# subtracting here to maintain score from bot0's perspective
				swing -= run_match(brains, bot1, bot0)
				# count scores
				scores[bot0] += swing
				scores[bot1] -= swing
				print "%d vs %d : %d" % (bot0, bot1, swing)

		if generation % save_every == 0:
			best_brain, _ = max(zip(brains, scores), key=lambda tup: tup[1])
			best_brain.save(save_dir, suffix="_generation%d" % generation)

		# select winners for mutation
		brains = [Perceptron.mate(choose_brain(brains, scores), choose_brain(brains, scores)).mutate(mutation_rate) for _ in xrange(population - newrandom)]
		
		# add new random bots to population
		brains.extend([Perceptron(layers, random=True) for _ in xrange(newrandom)])
		generation += 1
