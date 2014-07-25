#!/usr/bin/python
import random
from rgkit import rg
from neuralnets import MultiLayerPerceptron

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

class ScoredMultiLayerPerceptron(MultiLayerPerceptron):

	def __init__(self, *args, **kwargs):
		super(ScoredMultiLayerPerceptron, self).__init__(*args, **kwargs)
		self._score = 0

	def reset_score(self):
		self._score = 0

	def add_score(self, delta):
		self._score += delta

	def get_score(self):
		return self._score

	def mutate(self, edit=0.001):
		super(ScoredMultiLayerPerceptron, self).mutate(edit)
		self.reset_score()
		return self

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
		return ScoredMultiLayerPerceptron(**self.__bka)

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
			weight = random.random() * sum(weights)
			idx = sum(np.array(weights) < weight)
			return self.brains[idx]

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
		next = ScoredMultiLayerPerceptron.mate(self.choose_brain(), self.choose_brain()).mutate(self._mutate)
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
	def __init__(self, pop0, pop1, game_seed, n_games, mapfile=None):
		self.red_team = pop0
		self.blu_team = pop1

		self.__game_options  = Options(game_seed=game_seed, n_of_games=n_games, map_filepath=mapfile, print_info=True)

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
		r = Runner(players=[red_player, blu_player], options=self.__game_options)
		r.run()

def create_population(typ, size, rand, mut, seedfile, saveto, gen_length, slate_random):
	seed = []
	if seedfile:
		with open(seedfile, "r") as f:
			brainfiles = [ln.strip() for ln in f.readlines()]
			seed = [ScoredMultiLayerPerceptron.load(b) for b in brainfiles]
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

	tournament = Tournament(red, blu, game_seed, num_games, mapfile="rgkit/maps/minimap.py")
	tournament.run()
