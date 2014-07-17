#!/usr/bin/python
import os
import random
import numpy as np
from   math  import exp
from   rgkit import rg
# these imports are needed for running the evolution tournament
from rgkit.run import Options, Runner
from rgkit.game import Player
from rgkit.settings import settings as game_settings

def feature_display(feature_vector):
	from termcolor import cprint
	for x in range(5):
		for y in range(5):
			block_index = 4 * (5 * x + y)
			bgcolor = "on_white"
			txcolor  = "blue"
			ally_hp  = feature_vector[block_index + 0]
			enemy_hp = feature_vector[block_index + 1]
			spawn    = feature_vector[block_index + 2]
			obstacle = feature_vector[block_index + 3]
			content = "  "
			if ally_hp > 0:
				txcolor = "green"
				content = "%2d" % (ally_hp * rg.settings.robot_hp)
			elif enemy_hp > 0:
				txcolor = "red"
				content = "%2d" % (enemy_hp * rg.settings.robot_hp)
			if spawn > 0:
				bgcolor = "on_grey"
			if obstacle > 0:
				content = "##"
			cprint(" "+content+" ", txcolor, bgcolor, end="")
		print ""

def rand_plus_or_minus():
	return (random.random() * 2.0) - 1.0

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
	ens = []
	for pos in locs_around_55(bot.location):
		if pos in game.robots and game.robots[pos].player_id != bot.player_id:
			ens.append(game.robots[pos])
	return ens

@np.vectorize
def sigmoid(x):
	return 1.0 / (1.0 + exp(-x))

@np.vectorize
def sigmoid_derivy(y):
	return y * (1.0 - y)

@np.vectorize
def sigmoid_deriv(x):
	return np.multiply(sigmoid(x), 1.0 - sigmoid(x))

class Perceptron(object):

	def __init__(self, layer_sizes=[], mats=None, random=0.0):
		self.__W = mats or []
		self.__layers = len(layer_sizes)
		self.__layer_sizes = layer_sizes
		self.__reset_activations()
		if not mats:
			for i in xrange(self.__layers - 1):
				fromsize = layer_sizes[i]
				tosize = layer_sizes[i+1]
				mat = np.matrix(np.zeros((tosize, fromsize)))
				self.__W.append(mat)
		self.mutate(edit=random)

	def __str__(self):
		return "\n".join([str(mat) for mat in self.__W])

	def __reset_activations(self):
		# activations
		self.__a = map(lambda s: np.zeros((s,1)), self.__layer_sizes)
		# sum stage (a = sigmoid(z))
		self.__z  = map(lambda s: np.zeros((s,1)), self.__layer_sizes)

	def copy(self):
		# using __class__ allows subclasses to be copied also
		c = self.__class__(self.__layer_sizes)
		for i in range(self.__layers - 1):
			c.__W[i] = self.__W[i].copy()
		return c

	def mutate(self, edit=0.001):
		for mat in self.__W:
			changes = np.matrix([edit*rand_plus_or_minus() for _ in xrange(mat.size)])
			changes = changes.reshape(mat.shape)
			mat += changes
		return self

	def set_input(self, which, value):
		self.__a[0][which] = value

	def set_inputs(self, input_vector):
		self.__a[0] = input_vector

	def get_inputs(self):
		return self.__a[0].copy()

	def propagate(self):
		for l in xrange(self.__layers-1):
			self.__z[l+1] = self.__W[l] * self.__a[l]
			self.__a[l+1] = sigmoid(self.__z[l+1])

	def backpropagate(self, target, learning_rate=.2):
		"""Live update of matrix weights.
			'target' close to 1.0 reinforce (and 0.0 alter) behavior
		"""
		#########
		# step 1: compute errors at each layer
		#########
		errors = [None] * self.__layers
		L = self.__layers - 1 # L is highest layer index
		# error in last layer is defined by targets
		errors[L] = np.multiply(self.__a[L] - target, sigmoid_derivy(self.__a[L]))
		# propagate error backwards through the network
		for l in range(L-1, 0, -1):
			errors[l] = np.multiply(self.__W[l].transpose() * errors[l+1], sigmoid_derivy(self.__a[l]))
		#########
		# step 2: update matrix weights based on errors
		#########
		for l in range(L):
			self.__W[l] -= learning_rate * (errors[l+1] * self.__a[l].transpose())

	def get_output(self, which):
		return self.__a[-1][which]

	def choose_output(self, strict=False):
		if strict:
			maxi = 0
			maxv = self.__a[-1][0]
			for i in xrange(1, self.__layer_sizes[-1]):
				if self.__a[-1][i] > maxv:
					maxv = self.__a[-1].item(i)
					maxi = i
			return maxi, maxv
		else:
			summed = self.__a[-1].cumsum()
			index = random.random() * summed.item(-1)
			for i in xrange(0, self.__layer_sizes[-1]):
				if index <= summed.item(i):
					return i, self.__a[-1].item(i)
			return i, self.__a[-1].item(i)

	def get_output_vector(self):
		return self.__a[-1].copy()

	@classmethod
	def mate(cls, p0, p1):
		assert p0.__layer_sizes == p1.__layer_sizes
		newp = cls(layer_sizes=p0.__layer_sizes)
		for l in xrange(newp.__layers - 1):
			# select each weight randomly from either parent
			rows, cols = newp.__W[l].shape
			for r in xrange(rows):
				for c in xrange(cols):
					newp.__W[l][(r,c)] = random.choice([p0.__W[l][(r,c)], p1.__W[l][(r,c)]])
		return newp

	@classmethod
	def load(cls, file_path):
		p = None
		with open(file_path, "r") as f:
			npzdata = np.load(f)
			sizes = npzdata['arr_0']
			p = cls(sizes, random=False)
			for l in xrange(p.__layers - 1):
				p.__W[l] = np.matrix(npzdata['arr_%d' % (l+1)])
		return p

	def save(self, directory, suffix=""):
		fname = "perceptron_" + ("_".join([str(l) for l in self.__layer_sizes])) + suffix + ".npz"
		path = os.path.join(directory, fname)
		with open(path, "w") as f:
			mats_to_save = [np.array(self.__layer_sizes)] + self.__W
			np.savez(f, *mats_to_save)
		return path

class ScoredPerceptron(Perceptron):

	def __init__(self, *args, **kwargs):
		super(ScoredPerceptron, self).__init__(*args, **kwargs)
		self._score = 0

	def reset_score(self):
		self._score = 0

	def add_score(self, delta):
		self._score += delta

	def get_score(self):
		return self._score

	def mutate(self, edit=0.001):
		super(ScoredPerceptron, self).mutate(edit)
		self.reset_score()
		return self

class RobotPopulation(object):

	def __init__(self, pop):
		self.population = pop

	def population_init(self):
		if not hasattr(self, 'last_update'):
			self.last_update = -1
			# map from robot id to state from previous frame
			self.tracking_bots = {}

	def population_update(self, game):
		if self.last_update < game.turn:
			self.last_update = game.turn
			self.population.on_turn(game.turn)
			living_bots = [bot.robot_id for bot in game.robots.values() if bot.player_id == self.player_id]
			# remove dead from tracking
			delete_list = []
			for r_id, state in self.tracking_bots.iteritems():
				if r_id not in living_bots:
					delete_list.append(r_id)
			for r_id in delete_list:
				# debugging
				state = self.tracking_bots[r_id]
				# print "bot died"
				# feature_display(state["last_senses"])
				# print "having chosen", state["last_choice"]
				# print "from", state["last_output"]
				# raw_input()
				target_output = state["last_output"]
				target_output[state["last_choice"]] = 0.0
				self.population.group_lesson(state["last_senses"], target_output)
				self.population.remove(state["brain"])
				del self.tracking_bots[r_id]

			# add new brains
			for loc, bot in game.robots.iteritems():
				if bot.player_id == self.player_id and bot.robot_id not in self.tracking_bots:
					self.tracking_bots[bot.robot_id] = {
						"last_hp" : bot.hp,
						"brain": self.population.next_brain(),
						"enemies" : [],
						"last_senses": None,
						"last_output": None,
						"last_choice": None
					}

class Robot(RobotPopulation):

	width = 5
	halfwidth = width / 2
	inputs = (width ** 2) * 4 # 4 = {ally hp, enemy hp, spawn, obstacle}
	outputs = 10 # 4 attack, 4 move, guard, suicide
	layers = [width**2]
	selfishness = 1.0

	def act(self, game):
		# population_init has an effect once at the beginning of the match
		self.population_init()
		
		# add next generation and remove dead (once per turn)
		self.population_update(game)
		
		# for this specific robot, learn from previous iteration
		self.hindsight(game)
		
		senses = self.sense_environment(game)

		state = self.tracking_bots[self.robot_id]
		brain = state["brain"]

		# to encourage longevity, brain's scores are the sum of their health over time
		# (scores are used in selecting parents of the next generation)
		brain.add_score(self.hp)

		brain.set_inputs(senses)
		brain.propagate()

		index, value = brain.choose_output(strict=True)
		state["last_senses"] = senses
		state["last_output"] = brain.get_output_vector()
		state["last_choice"] = index
		state["enemies"]     = get_enemies_around(self, game)

		x,y = self.location
		choices = [
			['move', (x-1, y)],   ['move', (x+1, y)],   ['move', (x, y-1)],   ['move', (x, y+1)],
			['attack', (x-1, y)], ['attack', (x+1, y)], ['attack', (x, y-1)], ['attack', (x, y+1)],
			['guard'], ['suicide']]
		blind_choice = choices[index]
		if not action_is_valid(blind_choice):
			# negative here is a flag indicating that the last choice was invalid
			state["last_choice"] = -1 - state["last_choice"]
			return ['guard']
		return blind_choice

	def sense_environment(self, game):
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

	def hindsight(self, game):
		"""
		reinforce good habits, unlearn bad habits
		"""
		bot_state = self.tracking_bots[self.robot_id]
		enemies     = bot_state["enemies"]

		brain       = bot_state["brain"]
		last_senses = bot_state["last_senses"]
		last_output = bot_state["last_output"]
		last_choice = bot_state["last_choice"]

		if last_output is not None:
			ideal_output = last_output.copy()

			if last_choice < 0:
				# previous choice was simply invalid
				last_choice = -1 * (last_choice + 1)
				# ideal output in that case should have been zero if it was invalid, regardless of score
				ideal_output[last_choice] = 0.0
			else:
				# get team score
				global_allies  = [bot for bot in game.robots.values() if bot.player_id == self.player_id]
				global_enemies = [bot for bot in game.robots.values() if bot.player_id != self.player_id]
				team_score = float(len(global_allies) - len(global_enemies)) / rg.settings.spawn_per_player
				# get personal score
				personal_score = 0.0
				new_enemies = get_enemies_around(self, game)
				if len(enemies) > 0 and len(new_enemies) > 0:
					enemy_hp     = float(sum((en["hp"] for en in enemies)))     / len(enemies)
					new_enemy_hp = float(sum((en["hp"] for en in new_enemies))) / len(new_enemies)
					damage_dealt = enemy_hp - new_enemy_hp
					damage_felt  = bot_state["last_hp"] - self.hp
					personal_score = float(damage_dealt - damage_felt) / rg.settings.attack_range[1]
				# update ideal output based on whether score was good or bad
				s = Robot.selfishness
				ideal_output[last_choice] += sigmoid(personal_score * s + team_score * (1-s)) - 0.5
				ideal_output[last_choice] = max(min(ideal_output[last_choice], 1.0), 0.0)
				# debugging
				# if personal_score != 0:
				# 	print self.location
				# 	feature_display(last_senses)
				# 	print "with outputs (choice %d)" % last_choice, last_output
				# 	print "and result"
				# 	feature_display(self.sense_environment(game))
				# 	print "--> personal score of %f" % personal_score
				# 	print "--> ideal outputs of", ideal_output
				# 	raw_input()
			brain.set_inputs(last_senses)
			brain.propagate()
			brain.backpropagate(ideal_output, learning_rate=0.1)

class Population(object):

	def __init__(self, init_size, p_new_random=0.05, seed_pop=[], save_dir="", gen_length=0, **bka):
		self.__bka = bka
		# make sure randomness is part of blank brain construction
		self.__bka["random"] = self.__bka.get("random", 0.1)
		self.__size = init_size
		self.__p_new_random = p_new_random
		self.__save_dir = save_dir
		self.__gen_length = gen_length
		self._all_time_best = None
		self.brains = (seed_pop + [self.blank_slate() for _ in xrange(init_size)])[0:init_size]
		self.generation = 0

	def group_lesson(self, inputs, outputs, rate=0.5):
		for brain in self.brains:
			brain.set_inputs(inputs)
			brain.propagate()
			brain.backpropagate(outputs, learning_rate=rate)

	def blank_slate(self):
		return ScoredPerceptron(**self.__bka)

	def save_best(self, dest, extra=""):
		best_brain = self.choose_brain(strict=True)
		best_brain.save(dest, extra)

	def next_generation(self):
		self.generation += 1
		self.brains = [None] * self.__size
		for b in xrange(self.__size):
			if random.random() < self.__p_new_random:
				self.brains[b] = self.blank_slate()
			else:
				self.brains[b] = self.next_brain()

	def get_generation(self):
		return self.generation

	def remove(self, brain):
		if brain in self.brains:
			if self._all_time_best is None or brain.get_score() > self._all_time_best.get_score():
				self._all_time_best = brain
			self.brains.remove(brain)
		else:
			pass
			#print "could not find brain for removal"

	def choose_brain(self, strict=False, all_time=False):
		if len(self.brains) == 0 or all_time:
			return self._all_time_best
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

	def on_turn(self, turn):
		if (turn + 1) % 500 == 0 and self.__save_dir != "":
			self.save_best(self.__save_dir, "_turn%d_gen%d" % (turn, self.generation))
		if self.__gen_length > 0 and (turn+1) % self.__gen_length == 0:
			self.next_generation()

class Individuals(Population):
	"""Mutate each brain in isolation
	"""
	def __init__(self, size, p_new_random, mutation_rate, seed_pop=[], save_dir="", gen_length=0, **brain_kwargs):
		super(Individuals, self).__init__(size, p_new_random, seed_pop, save_dir, gen_length, **brain_kwargs)
		self._mutate = mutation_rate

	def next_brain(self):
		next = self.choose_brain().mutate(self._mutate)
		self.brains.append(next)
		return next

class Family(Population):
	"""Cross together fittest from population
	"""
	def __init__(self, size, p_new_random, mutation_rate, seed_pop=[], save_dir="", gen_length=0, **brain_kwargs):
		super(Family, self).__init__(size, p_new_random, seed_pop, save_dir, gen_length, **brain_kwargs)
		self._mutate = mutation_rate

	def next_brain(self):
		next = ScoredPerceptron.mate(self.choose_brain(), self.choose_brain()).mutate(self._mutate)
		self.brains.append(next)
		return next

class Collective(Population):
	"""All bots share the same few brains
		(population never grows larger than the initial size)
	"""
	def __init__(self, size, p_new_random, mutation_rate, seed_pop=[], save_dir="", gen_length=0, **brain_kwargs):
		super(Collective, self).__init__(size, p_new_random, seed_pop, save_dir, gen_length, **brain_kwargs)
		self._mutate = mutation_rate

	def next_brain(self):
		return self.choose_brain(strict=False)

	def remove(self, brain):
		"""override Population.remove since this population has an unchanging selection of brains
		"""
		if brain in self.brains:
			if self._all_time_best is None or brain.get_score() > self._all_time_best.get_score():
				self._all_time_best = brain


class Tournament(object):
	def __init__(self, pop0, pop1, game_seed, n_games, turns_per_game):
		self.red_team = pop0
		self.blu_team = pop1

		self.__game_options  = Options(game_seed=game_seed, n_of_games=n_games, print_info=True)
		self.__game_settings = game_settings
		self.__game_settings['max_turns'] = turns_per_game

	def get_1v1_matches(self):
		for r in self.red_team.brains:
			for b in self.blu_team.brains:
				yield (r, b)

	def next_generation(self):
		self.red_team.next_generation()
		self.blu_team.next_generation()

	def run(self):
		red_robot  = Robot(self.red_team)
		red_player = Player(robot=red_robot, name="Red")
		blu_robot  = Robot(self.blu_team)
		blu_player = Player(robot=blu_robot, name="Blu")
		r = Runner(players=[red_player, blu_player], options=self.__game_options, settings=self.__game_settings)
		r.run()

def create_population(typ, size, rand, mut, seedfile, saveto, gen_length, slate_random):
	seed = []
	if seedfile:
		with open(seedfile, "r") as f:
			brainfiles = [ln.strip() for ln in f.readlines()]
			seed = [ScoredPerceptron.load(b) for b in brainfiles]
	args = [size, rand, mut, seed, saveto, gen_length]
	brain_kwargs = {
		"layer_sizes" : [Robot.inputs] + Robot.layers + [Robot.outputs],
		"random" : slate_random
	}
	if typ.lower() == "individuals":
		return Individuals(*args, **brain_kwargs)
	elif typ.lower() == "family":
		return Family(*args, **brain_kwargs)
	elif typ.lower() == "collective":
		return Collective(*args, **brain_kwargs)
	else:
		print "unknown population type: %s" % typ

if __name__ == '__main__':
	from evo_options import *

	if game_seed:
		random.seed(game_seed)

	red = create_population(pop0, pop0_size, pop0_random, pop0_mutation, pop0_seed, pop0_saveto, pop0_gen_length, pop0_slate_rand)
	blu = create_population(pop1, pop1_size, pop1_random, pop1_mutation, pop1_seed, pop1_saveto, pop1_gen_length, pop1_slate_rand)

	tournament = Tournament(red, blu, game_seed, num_games, turns_per)
	tournament.run()
