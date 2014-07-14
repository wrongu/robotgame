#!/usr/bin/python
import os
import re
import random
import numpy as np
from   math  import exp
from   rgkit import rg

def rand_plus_or_minus():
	return (random.random() * 2.0) - 1.0

class Perceptron(object):

	def __init__(self, layer_sizes=[], mats=None, random=False, horizontal=False):
		self.__W = mats or []
		self.__layers = len(layer_sizes)
		self.__sizes = layer_sizes
		self.__horizontal = horizontal
		self.__reset_activations()
		if not mats:
			for i in range(len(layer_sizes) - 1):
				fromsize = layer_sizes[i]
				tosize = layer_sizes[i+1]
				mat = np.matrix(np.zeros((tosize, fromsize)))
				self.__W.append(mat)
		if random:
			self.mutate(edit=1.0)

	def __str__(self):
		return "\n".join([str(mat) for mat in self.__W])

	def __reset_activations(self):
		# activations
		self.__a = map(lambda s: np.zeros((s,1)), self.__sizes)
		# sum stage (a = sigmoid(z))
		self.__z  = map(lambda s: np.zeros((s,1)), self.__sizes)

	@staticmethod
	@np.vectorize
	def sigmoid(x):
		return 1.0 / (1.0 + exp(-x))

	@staticmethod
	@np.vectorize
	def sigmoid_derivy(y):
		return y * (1.0 - y)

	@staticmethod
	@np.vectorize
	def sigmoid_deriv(x):
		return np.multiply(Perceptron.sigmoid(x), 1.0 - Perceptron.sigmoid(x))

	def copy(self):
		c = self.__class__(self.__sizes)
		for mat in self.__W:
			c.__W.append(mat.copy())
		return c

	def mutate(self, edit=0.001):
		for mat in self.__W:
			changes = np.matrix([edit*rand_plus_or_minus() for _ in xrange(mat.size)])
			changes = changes.reshape(mat.shape)
			mat += changes
		return self

	def set_input(self, which, value):
		assert -1 <= value <= 1
		assert 0 <= which < self.__sizes[0]
		self.__a[0][which] = value

	def set_inputs(self, input_vector):
		self.__a[0] = input_vector

	def propagate(self, iterations=1):
		if not self.__horizontal: iterations = 1
		for i in xrange(iterations):
			for l in xrange(self.__layers-1):
				self.__z[l+1]  = self.__W[l] * self.__a[l]
				self.__a[l+1] = Perceptron.sigmoid(self.__z[l+1])

	def backpropagate(self, target, learning_rate=.2):
		"""Live update of matrix weights.
			'target' close to 1.0 reinforce (and 0.0 alter) behavior
		"""
		# e = target - self.__a[-1]
		# s_d = Perceptron.sigmoid_derivy(self.__a[-1])
		# delta = np.multiply(e, s_d) * self.__a[-2].transpose()
		# self.__W[-1] += delta * learning_rate
		#########
		# step 1: compute errors at each layer
		#########
		errors = [None] * self.__layers
		L = self.__layers - 1 # L is highest layer index
		# error in last layer is defined by targets
		errors[L] = np.multiply(self.__a[L] - target, Perceptron.sigmoid_derivy(self.__a[L]))
		# propagate error backwards through the network
		for l in range(L-1, 0, -1):
			errors[l] = np.multiply(self.__W[l].transpose() * errors[l+1], Perceptron.sigmoid_derivy(self.__a[l]))
		#########
		# step 2: update matrix weights based on errors
		#########
		for l in range(L):
			self.__W[l] -= learning_rate * (errors[l+1] * self.__a[l].transpose())

	def get_output(self, which):
		assert 0 <= which < self.__sizes[-1]
		return self.__a[-1][which]

	def choose_output(self, strict=False):
		if strict:
			maxi = 0
			maxv = self.__a[-1][0]
			for i in xrange(1, self.__sizes[-1]):
				if self.__a[-1][i] > maxv:
					maxv = self.__a[-1][i]
					maxi = i
			return maxi, maxv
		else:
			summed = self.__a[-1].cumsum()
			index = random.random() * summed.item(-1)
			for i in xrange(0, self.__sizes[-1]):
				if index <= summed.item(i):
					return i, self.__a[-1][i]
			return i, self.__a[-1][i]

	def get_output_vector(self):
		return self.__a[-1]

	@classmethod
	def mate(cls, p0, p1):
		assert p0.__sizes == p1.__sizes
		assert p0.__horizontal == p1.__horizontal
		newp = cls(layer_sizes=p0.__sizes, horizontal=p0.__horizontal)
		for l in xrange(len(newp.__sizes)-1):
			rows, cols = newp.__W[l].shape
			for r in xrange(rows):
				for c in xrange(cols):
					newp.__W[l][(r,c)] = random.choice([p0.__W[l][(r,c)], p1.__W[l][(r,c)]])
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
			p.__W = [np.matrix(s) for s in matstrings]
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

def locs_around_55(loc):
	xc,yc = loc
	for x in xrange(xc-2,xc+3):
		for y in xrange(yc-2,yc+3):
			typ = rg.loc_types((x,y))
			if "invalid" not in typ and "obstacle" not in typ:
				yield (x,y)
def get_enemies_around(bot, game):
	for pos in locs_around_55(bot.location):
		if pos in game.robots and game.robots[pos].player_id != bot.player_id:
			yield game.robots[pos]

class RobotPopulation(object):

	width = 5
	halfwidth = width / 2
	inputs = (width ** 2) * 4 # 4 = {ally hp, enemy hp, spawn, obstacle}
	outputs = 9 # 4 attack, 4 move, guard, suicide (suicide removed temporarily)
	layers = [width**2]
	selfishness = 1.0

	def __init__(self, pop):
		self.population = pop

	def population_init(self):
		if not hasattr(self, 'bot_states'):
			# map from robot id to state, where state has 'brain', 'bot', 'last_hp', 'enemies', 'last_output', and 'last_choice'
			self.bot_states = {}
			self.last_update = -1

	def population_update(self, game):
		if self.last_update < game.turn:
			self.last_update = game.turn
			bots_by_id = {bot.robot_id: bot for bot in game.robots.values() if bot.player_id == self.player_id}
			if (game.turn + 1) % 100 == 0:
				self.population.save_best("brains55", extra="_pop%d_turn%5d" % (self.player_id, game.turn+1))
				self.population.next_generation()
			else:
				enemies = [bot for bot in game.robots.values() if bot.player_id != self.player_id]
				team_score = (len(bots_by_id) - len(enemies)) / rg.settings.spawn_per_player
				# learn from last frame
				delete_list = []
				for r_id, state in self.bot_states.iteritems():
					brain = state["brain"]
					prev_output = state["last_output"]
					prev_choice = state["last_choice"]
					bot = state["bot"]
					enemies = state["enemies"]
					# output is None if the robot was just spawned
					if prev_output is not None:
						# here we calculate the effectiveness of the previous decision,
						# then run the learning algorithm to enforce or punish that behaviour
						prev_enemy_hp = sum(b.hp for b in enemies)
						state["enemies"] = get_enemies_around(bot, game)
						enemies = state["enemies"]
						curr_enemy_hp = sum(b.hp for b in enemies)
						# this should enforce fleeing or attacking. positive score when enemies lose health
						selfish_score = float(prev_enemy_hp - curr_enemy_hp) / float(rg.settings.attack_range[1])
						if r_id not in bots_by_id:
							# if died, detract from selfish score
							selfish_score -= 1.0
							delete_list.append(r_id)
						else:
							# still alive. selfish score is good if health was retained
							new_hp  = bots_by_id[r_id]["hp"]
							selfish_score += (new_hp - state["last_hp"]) / rg.settings.robot_hp
							state["last_hp"] = new_hp
						s = RobotPopulation.selfishness
						decision_score = Perceptron.sigmoid(selfish_score * s + team_score * (1-s))
						brain.update_score(decision_score)
						prev_output[prev_choice] += decision_score - 0.5
						brain.backpropagate(prev_output, learning_rate=0.8)
				for r_id in delete_list:
					del self.bot_states[r_id]

			# add new brains
			for loc, bot in game.robots.iteritems():
				if bot.player_id == self.player_id and bot.robot_id not in self.bot_states:
					self.bot_states[bot.robot_id] = {
						"last_hp" : bot.hp,
						"brain": self.population.next_brain(),
						"bot" : bot,
						"enemies" : [],
						"last_output": None,
						"last_choice": None
					}

	def construct_features(self, game):
		xc, yc = self.location
		features = np.zeros((RobotPopulation.inputs,1))
		i = 0
		for x in xrange(xc - RobotPopulation.halfwidth, xc + RobotPopulation.halfwidth + 1):
			for y in xrange(yc - RobotPopulation.halfwidth, yc + RobotPopulation.halfwidth + 1):
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

class Robot(RobotPopulation):

	def act(self, game):
		self.population_init()
		self.population_update(game)
		
		x,y = self.location

		features = self.construct_features(game)

		state = self.bot_states[self.robot_id]
		brain = state["brain"]

		brain.set_inputs(features)
		brain.propagate()

		index, value = brain.choose_output(strict=False)
		self.bot_states[self.robot_id]["last_output"] = brain.get_output_vector()
		self.bot_states[self.robot_id]["last_choice"] = index

		choices = [
			['move', (x-1, y)],   ['move', (x+1, y)],   ['move', (x, y-1)],   ['move', (x, y+1)],
			['attack', (x-1, y)], ['attack', (x+1, y)], ['attack', (x, y-1)], ['attack', (x, y+1)],
			['guard']]
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
		return self

class Population(object):

	def __init__(self, size, new_random, **bka):
		self.__bka = bka
		self.__size = size
		self.__new_random = new_random
		self.generation = 0

	def make_brain(self):
		return ScoredPerceptron(**self.__bka)

	def save_best(self, dest, extra=""):
		best_brain = max(self.brains, key=lambda b: b.get_score())
		best_brain.save(dest, "%s_generation%d" % (extra, self.generation))

	def next_generation(self):
		self.generation += 1
		self.brains = [self.next_brain() for _ in xrange(self.__size - self.__new_random)]
		self.brains.extend([self.make_brain() for _ in xrange(self.__new_random)])

	def get_generation(self):
		return self.generation

	def remove_brain(self, brain):
		if brain in self.brains:
			self.brains.remove(brain)
		else:
			print "could not find brain for removal"

	def choose_brain(self, strict=False):
		if strict:
			return max(self.brains, key=lambda brain: brain.get_score())
		else:
			# weighted choice based on score
			scores = [brain.get_score() for brain in self.brains]
			divisor = max([abs(s) for s in scores]) + 1.0
			weights = [1.0 / (1 + exp(-s / divisor)) for s in scores]
			idx = random.random() * sum(weights)
			s = 0.0
			for i in xrange(len(self.brains)):
				s += weights[i]
				if s >= idx:
					return self.brains[i]

class Individuals(Population):
	"""Mutate each brain in isolation
	"""
	def __init__(self, size, new_random, mutation_rate, seed_pop=[], **brain_kwargs):
		super(Individuals, self).__init__(size, new_random, **brain_kwargs)
		self.brains = (seed_pop + [self.make_brain() for _ in xrange(size)])[0:size]
		self._mutate = mutation_rate

	def next_brain(self):
		next = self.choose_brain().mutate(self._mutate)
		self.brains.append(next)
		return next

class Family(Population):
	"""Cross together fittest from population
	"""
	def __init__(self, size, new_random, mutation_rate, seed_pop=[], **brain_kwargs):
		super(Family, self).__init__(size, new_random, **brain_kwargs)
		self.brains = (seed_pop + [self.make_brain() for _ in xrange(size)])[0:size]
		self._mutate = mutation_rate

	def next_brain(self):
		next = ScoredPerceptron.mate(self.choose_brain(), self.choose_brain()).mutate(self._mutate)
		self.brains.append(next)
		return next

class Tournament(object):
	def __init__(self, population_class, team_size, new_random, mutation_rate, **brain_kwargs):
		self.red_team = population_class(team_size, new_random, mutation_rate, **brain_kwargs)
		self.blu_team = population_class(team_size, new_random, mutation_rate, **brain_kwargs)

	def get_matches(self):
		for r in self.red_team.brains:
			for b in self.blu_team.brains:
				yield (r, b)

	def next_generation(self):
		self.red_team.next_generation()
		self.blu_team.next_generation()

	def save_best(self, dest, extra=""):
		self.red_team.save_best(dest, "_red"+extra)
		self.blu_team.save_best(dest, "_blu"+extra)

# tournament
if __name__ == '__main__':
	from rgkit.run import Options, Runner
	from rgkit.game import Player
	from rgkit.settings import settings as game_settings
	from optparse import OptionParser

	parser = OptionParser()
	parser.add_option("-s", "--seed",          type="int",   dest="seed",         default=0,    help="set seed for games' RNG")
	parser.add_option("-g", "--generations",   type="int",   dest="generations",  default=500,  help="number of generations")
	parser.add_option("-n", "--ngames",        type="int",   dest="ngames",       default=1,    help="number of games per generation")
	parser.add_option("-p", "--population",    type="int",   dest="population",   default=8,    help="population size")
	parser.add_option("-r", "--random",        type="int",   dest="random",       default=1,    help="number of random brains to add each generation")
	parser.add_option("-e", "--save-every",    type="int",   dest="save_every",   default=100,  help="generations between saves")
	parser.add_option("-T", "--turns",         type="int",   dest="turns",        default=100,  help="turns per game")
	parser.add_option("-m", "--mutation-rate", type="float", dest="mutationrate", default=0.01, help="mutation rate")
	parser.add_option("-o", "--output-dir",    type="string",dest="savedir",      default="brains", help="directory to save brains")
	parser.add_option("-t", "--pop-type",      type="string",dest="pop_type",     default="Family", help="population type (Individual, Family, Team)")
	parser.add_option("-l", "--load-file",     type="string",dest="load_file",     help="path to a file containing paths to brains on each line. This is an alternative to naming each file in the args")
	(options, leftover_args) = parser.parse_args()

	pop_size = options.population
	new_random = options.random
	ngames = options.ngames
	mutation_rate = options.mutationrate
	max_generation = options.generations
	rand_seed = options.seed
	save_dir = options.savedir
	save_every = options.save_every
	game_settings["max_turns"] = options.turns

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
		"layer_sizes" : [RobotPopulation.inputs] + RobotPopulation.layers + [RobotPopulation.outputs],
		"random" : True
	}
	if options.pop_type:
		if options.pop_type.lower() == "individuals":
			population = Individuals
		elif options.pop_type.lower() == "family":
			population = Family
		else:
			print "unkown population type '%s'" % options.pop_type

	tournament = Tournament(population, pop_size, new_random, mutation_rate, **brain_construction)

	def run_match(pop0, pop1):
		r0 = Robot(pop0)
		p0 = Player(robot=r0)
		r1 = Robot(pop1)
		p1 = Player(robot=r1)

		opts = Options(n_of_games=ngames, game_seed=pop0.get_generation() + rand_seed, print_info=True)
		r = Runner(players=[p0, p1], options=opts, settings=game_settings)
		results = r.run()
		swing = 0
		for res in results:
			swing += res[0] - res[1]
		return swing

	g = 0
	while g < max_generation:
		g += 1
		print "--- generation %d ---" % g

		# run 1v1 matches
		run_match(tournament.red_team, tournament.blu_team)

		if save_every != 0 and g % save_every == 0:
			tournament.save_best(options.savedir)

		tournament.next_generation()
