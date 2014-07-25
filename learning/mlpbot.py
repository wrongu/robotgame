import numpy as np
from rgkit import rg
from neuralnets import MultiLayerPerceptron

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

class Robot:

	# sense configurations
	width = 5                 # width must be odd so sense range is centered on bot
	halfwidth = width / 2     # 'radius' of sensory range
	inputs = (width ** 2) * 4 # {sensory range} * {ally-hp, enemy-hp, spawn, obstacle}
	outputs = 10              # 4 attack, 4 move, guard, suicide
	layers = [width**2] # heuristic: hidden layer as big as sensory range
	selfishness = 0.7         # [0,1]
	damage_multiplier = 2.0   # how much more strongly to weight "damage_felt" than "damage_dealt"

	def __init__(self, online_learning=True):
		self._online_learning = online_learning

	def match_init(self):
		if not hasattr(self, 'recall'):
			self.recall = {} # map from robot id to last-turn info

	def next_brain(self):
		return MultiLayerPerceptron.load("mlp_100_25_10incubated_400_0.020000.npz")
		# return MultiLayerPerceptron(layer_sizes=[Robot.inputs] + Robot.layers + [Robot.outputs], random=0.1)

	def override_next_brain(self, brain_factory):
		Robot.next_brain = brain_factory

	def match_update(self, game):
		for loc, bot in game.robots.iteritems():
			if bot.player_id == self.player_id:
				# check if new
				if bot.robot_id not in self.recall:
					self.recall[bot.robot_id] = {
						'brain' : self.next_brain()
					}

	def act(self, game):
		# match_init has an effect once per match
		self.match_init()
		# match_update has an effect once per turn
		self.match_update(game)
		
		state = self.recall[self.robot_id]

		# for this specific robot, learn from previous iteration
		if self._online_learning:
			self.hindsight(game)
		
		# sense environment and act
		senses = self.sense_environment(game)

		brain = state["brain"]
		brain.set_input_vector(senses)
		brain.propagate()

		index = brain.choose_output(strict=True)
		state["last_senses"] = senses
		state["last_output"] = brain.get_output_vector()
		state["last_choice"] = index
		state["enemies"]     = get_enemies_around(self, game)
		state["last_hp"]     = self.hp

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
		bot_state   = self.recall[self.robot_id]
		if "last_output" not in bot_state:
			# bot just spawned, nothing to learn...
			return
		brain       = bot_state["brain"]
		enemies     = bot_state["enemies"]
		last_hp     = bot_state["last_hp"]
		last_senses = bot_state["last_senses"]
		last_output = bot_state["last_output"]
		last_choice = bot_state["last_choice"]

		ideal_output = last_output.copy()
		learn = False

		if last_choice < 0:
			# previous choice was simply invalid
			last_choice = -1 * (last_choice + 1)
			# ideal output in that case should have been zero if it was invalid, regardless of score
			# print self.location,"did invalid"
			ideal_output[last_choice] = 0.0
			learn = True
		else:
			# get team score
			global_allies  = [bot for bot in game.robots.values() if bot.player_id == self.player_id]
			global_enemies = [bot for bot in game.robots.values() if bot.player_id != self.player_id]
			n_a = len(global_allies)
			n_e = len(global_enemies)
			tot = n_a + n_e
			team_score = float(n_a - n_e) / float(tot) if tot > 0 else 0.0
			# get personal score
			personal_score = 0.0
			new_enemies = get_enemies_around(self, game)
			if len(enemies) > 0 and len(new_enemies) > 0:
				avg_enemy_hp     = float(sum((en["hp"] for en in enemies)))     / len(enemies)
				new_avg_enemy_hp = float(sum((en["hp"] for en in new_enemies))) / len(new_enemies)
				damage_dealt = abs(avg_enemy_hp - new_avg_enemy_hp)
				damage_felt  = abs(last_hp - self.hp) * Robot.damage_multiplier
				personal_score = float(damage_dealt - damage_felt) / rg.settings.attack_range[1]
			# update ideal output based on whether score was good or bad
			s = Robot.selfishness
			net_score = personal_score * s + team_score * (1-s)
			learn = net_score != 0
			if net_score > 0:
				# reinforce good behavior
				ideal_output[last_choice] = 1.0
			elif net_score < 0:
				# unlearn bad behavior and boost others
				ideal_output = np.ones(ideal_output.shape)
				ideal_output[last_choice] = 0.0
			# print self.location, "did", ("good" if net_score > 0 else ("bad" if net_score < 0 else "neutral"))
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
		if learn:
			brain.set_input_vector(last_senses)
			brain.propagate()
			brain.backpropagate(ideal_output, learning_rate=0.2)

if __name__ == '__main__':
	from rgkit.run import Runner, Options
	from rgkit.game import Player
	from sys import argv
	if len(argv) > 1:
		brain0 = MultiLayerPerceptron.load(argv[1])
		brain1 = MultiLayerPerceptron.load(argv[1])

		robot0 = Robot(False)
		robot1 = Robot(False)

		robot0.override_next_brain(lambda s: brain0)
		robot1.override_next_brain(lambda s: brain1)

		player0 = Player(name="red", robot=robot0)
		player1 = Player(name="red", robot=robot1)

		opts = Options(print_info=True)
		r = Runner(players=[player0, player1], options=opts)
		r.run()