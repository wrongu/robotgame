#!/usr/bin/python
import os
import re
import random
import itertools
import numpy as np
from   math  import exp
from   rgkit import rg

def rand_plus_or_minus():
	return (random.random() * 2.0) - 1.0

class Perceptron(object):

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

	@staticmethod
	@np.vectorize
	def sigmoid_deriv(s):
		return s * (1.0 - s)

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
				self.__states[l+1] = Perceptron.sigmoid(self.__weight_matrices[l] * self.__states[l])

	def backpropagate(self, target, learning_rate=.01):
		"""Live update of matrix weights.
			'target' close to 1.0 reinforce (and 0.0 alter) behavior
		"""
		e = target - self.__states[-1]
		s_d = Perceptron.sigmoid_deriv(self.__states[-1])
		delta = np.multiply(e, s_d) * self.__states[-2].transpose()
		self.__weight_matrices[-1] += delta * learning_rate

	def get_output(self, which):
		assert 0 <= which < self.__sizes[-1]
		return self.__states[-1][which]

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

	@classmethod
	def mate(cls, p0, p1):
		assert p0.__sizes == p1.__sizes
		assert p0.__horizontal == p1.__horizontal
		newp = cls(layer_sizes=p0.__sizes, horizontal=p0.__horizontal)
		for l in xrange(len(newp.__sizes)-1):
			rows, cols = newp.__weight_matrices[l].shape
			for r in xrange(rows):
				for c in xrange(cols):
					newp.__weight_matrices[l][(r,c)] = random.choice([p0.__weight_matrices[l][(r,c)], p1.__weight_matrices[l][(r,c)]])
		return newp

	@classmethod
	def load(cls, file_path):
		p = None
		with open(file_path, "r") as f:
			sizes = [int(n) for n in f.readline().split()]
			p = cls(sizes, random=False)
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

class Robot(object):

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

class ScoredPerceptron(Perceptron):

	def __init__(self, *args, **kwargs):
		super(ScoredPerceptron, self).__init__(*args, **kwargs)
		self._score = 0

	def reset_score(self):
		self._score = 0

	def update_score(self, delta):
		self._score += delta

	def get_score(self):
		return self._score

	def mutate(self, edit=0.001):
		super(ScoredPerceptron, self).mutate(edit)
		self.reset_score()

class Population(object):

	def __init__(self, *ba, **bka):
		self.__ba = ba
		self.__bka = bka
		self._generation = 0

	def make_brain(self):
		return ScoredPerceptron(*self.__ba, **self.__bka)

	def save_best(self, dest, extra=""):
		_, best_brain = max(zip(self._scores, self._brains))
		best_brain.save(dest, "%s_generation%d" % (extra, self._generation))

	def get_generation(self):
		return self._generation

	@staticmethod
	def choose_brain(brain_list, strict=False):
		if strict:
			return max(brain_list, key=lambda brain: brain.get_score())
		else:
			# weighted choice based on score
			scores = [brain.get_score() for brain in brain_list]
			weights = [1.0 / (1 + exp(-s / (max(scores)+0.1))) for s in scores]
			idx = random.random() * sum(weights)
			s = 0.0
			for i in xrange(len(brain_list)):
				s += weights[i]
				if s >= idx:
					return brain_list[i]

class Individuals(Population):
	"""Mutate each brain in isolation
	"""
	def __init__(self, size, new_random, mutation_rate, *brain_args, **brain_kwargs):
		super(Individuals, self).__init__(*brain_args, **brain_kwargs)
		self._brains = [self.make_brain() for _ in xrange(size)]
		self._new_random = new_random
		self._mutate = mutation_rate

	def get_matches(self):
		for i in xrange(len(self._brains)):
			for j in xrange(i+1, len(self._brains)):
				yield (self._brains[i], self._brains[j])

	def next_generation(self):
		self._generation += 1
		self._brains = [Population.choose_brain(self._brains).mutate(self._mutate) for _ in xrange(len(self._brains) - self._new_random)]
		self._brains.extend([self.make_brain() for _ in xrange(self._new_random)])

class Family(Population):
	"""Cross together fittest from population
	"""
	def __init__(self, size, new_random, mutation_rate, *brain_args, **brain_kwargs):
		super(Family, self).__init__(*brain_args, **brain_kwargs)
		self._brains = [self.make_brain() for _ in xrange(size)]
		self._new_random = new_random
		self._mutate = mutation_rate

	def get_matches(self):
		for i in xrange(len(self._brains)):
			for j in xrange(i+1, len(self._brains)):
				yield (self._brains[i], self._brains[j])

	def next_generation(self):
		self._generation += 1
		self._brains = [Perceptron.mate(
			self.choose_brain(self._brains),
			self.choose_brain(self._brains)).mutate(self._mutate)
			for _ in xrange(len(self._brains) - self._new_random)]
		self._brains.extend([self.make_brain() for _ in xrange(self._new_random)])

class Teams(Population):
	"""Two families against each other
	"""
	def __init__(self, team_size, new_random, mutation_rate, *brain_args, **brain_kwargs):
		super(Teams, self).__init__(brain_args, brain_kwargs)
		self._red_team = Family(team_size, new_random, mutation_rate)
		self._blu_team = Family(team_size, new_random, mutation_rate)

	def get_matches(self):
		for r in self._red_team._brains:
			for b in self._blu_team._brains:
				yield (r, b)

	def next_generation(self):
		self._generation += 1
		self._red_team.next_generation()
		self._blu_team.next_generation()

	def save_best(self, dest, extra=""):
		self._red_team.save_best(dest, "_red"+extra)
		self._red_team.save_best(dest, "_blu"+extra)

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
	parser.add_option("-t", "--pop-type",      type="string",dest="pop_type",     default="Family", help="population type (Individual, Family, Team)")
	parser.add_option("-l", "--load-file",     type="string",dest="load_file",     help="path to a file containing paths to brains on each line. This is an alternative to naming each file in the args")
	(options, leftover_args) = parser.parse_args()

	popsize = options.population
	newrandom = options.random
	ngames = options.ngames
	mutation_rate = options.mutationrate
	max_generation = options.generations
	rand_seed = options.seed
	save_dir = options.savedir
	save_every = 100

	random.seed(rand_seed)

	load_brains = []
	for arg in leftover_args:
		try:
			p = ScoredPerceptron.load(arg)
			if p is not None:
				load_brains.append(p)
		except:
			print "could not load brain from file '%s'" % (arg)

	if options.load_file:
		with open(options.load_file, 'r') as f:
			fnames = f.readlines()
		for f in fnames:
			try:
				p = ScoredPerceptron.load(f)
				if p is not None:
					load_brains.append(p)
			except:
				print "could not load brain from file '%s'" % (f)

	brain_construction = {
		"layer_sizes" : [Robot.inputs] + Robot.layers + [Robot.outputs]
	}

	population = Individuals(popsize, newrandom, mutation_rate, **brain_construction)
	if options.pop_type:
		if options.pop_type.lower() == "individuals":
			population = Individuals(popsize, newrandom, mutation_rate, **brain_construction)
		elif options.pop_type.lower() == "family":
			population = Family(popsize, newrandom, mutation_rate, **brain_construction)
		elif options.pop_type.lower() == "teams":
			population = Teams(popsize/2, newrandom, mutation_rate, **brain_construction)
		else:
			print "unkown population type '%s'" % options.pop_type

	def run_match(brain0, brain1):
		r0 = Robot()
		r0.set_brain(brain0)
		p0 = Player(robot=r0)
		r1 = Robot()
		r1.set_brain(brain1)
		p1 = Player(robot=r1)

		opts = Options(n_of_games=ngames, game_seed=population.get_generation() + rand_seed)
		r = Runner(players=[p0, p1], options=opts)
		results = r.run()
		swing = 0
		for res in results:
			swing += res[0] - res[1]
		brain0.update_score(swing)
		brain1.update_score(-swing)

	while population.get_generation() < max_generation:
		print "--- generation %d ---" % population.get_generation()

		# run 1v1 matches
		for (brain0, brain1) in population.get_matches():	
			# run 0 vs 1
			run_match(brain0, brain1)
			# run 1 vs 0 (left/right symmetry)
			run_match(brain1, brain0)

		if population.get_generation() % save_every == 0:
			population.save_best(options.savedir)
	
	# save at the end also
	population.save_best(options.savedir)

