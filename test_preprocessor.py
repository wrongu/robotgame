import rg, random

"""
Low-level actions:
 - guard
 - suicide
 - move
 - attack

High-level individual actions:
 - flee
 - panic
 - trap
 - attack
 - navigate

High-level team actions:
 - protect (defense)
 - swarm (offsense)
"""

def spawn_is_happening(game):
    return game['turn'] % 10 == 0

def around(loc):
	return rg.locs_around(loc, filter_out=("invalid", "obstacle"))

class ActionPriority:

	MIN = -1000

	def __init__(self, bot):
		self.robot = bot
		self.options = []

	def setMove(self, priority, target):
		self.options.append((priority, ['move', target]))

	def setAttack(self, priority, target):
		self.options.append((priority, ['attack', target]))

	def setSuicide(self, priority):
		self.options.append((priority, ['suicide']))
	
	def setGuard(self, priority):
		self.options.append((priority, ['guard']))

	def getAction(self, under):
		options = [option for option in self.options if option[0] < under]
		options.sort(reverse=True, key=lambda x: x[0])
		if len(options) > 0:
			return options[0]
		else:
			return (0, ['guard'])

class Robot:

	def act(self, game):
		is_init = self.setup_shared_data(game)
		if is_init:
			print "---- PREPROCESS ----"
			self.preprocess(game)
			print "---- CHOOSE ACTS ----"
			self.choose_actions(game)
		return self.decisions[self.location]

	###################
	# Per-robot logic #
	###################

	def get_action_priorities(self, robot, game):
		"""Like act(), but return an ActionPriority for this robot
		"""
		robots = game['robots']
		def is_enemy(loc):
			return (loc in robots and robots[loc].player_id != robot.player_id)

		def is_ally(loc):
			return (loc in robots and robots[loc].player_id == robot.player_id)

		def threat(loc):
			return sum([(1 if is_enemy(next) else 0) for next in around(loc)])

		def value(damage_deal, my_location, guard=False, suicide=False):
			# TODO keep track of whether traps have been set before by this opponent
			threat_count = threat(my_location)
			pain = threat_count * 10 if not guard else threat_count * 5
			if spawn_is_happening(game) and "spawn" in rg.loc_types(my_location):
				return -50
			elif not suicide:
				return damage_deal - pain
			else:
				return damage_deal - 50 if pain < robot.hp else -50

		ap = ActionPriority(robot)
		near = around(robot.location)
		# option 1: move
		moveable = [loc for loc in near if not is_enemy(loc)]
		adj_enemies = [loc for loc in near if is_enemy(loc)]
		random.shuffle(moveable)
		random.shuffle(adj_enemies)
		for next in moveable:
			ap.setMove(value(0, next), next)
		# option 2: attack
		for en_loc in adj_enemies:
			ap.setAttack(value(9, robot.location), en_loc)
		# option 3: guard
		ap.setGuard(value(0, robot.location, guard=True))
		# option 4: suicide
		ap.setSuicide(value(9*len(adj_enemies), robot.location, suicide=True))
		return ap

	########################
	# Once-per-frame logic #
	########################

	def preprocess(self, game):
		"""Get actions for all ally robots
		"""
		for loc, bot in game['robots'].items():
			if bot.player_id == self.player_id:
				self.pre_actions[loc] = self.get_action_priorities(bot, game)
		
	def choose_actions(self, game):
		for loc, bot in game['robots'].items():
			if bot.player_id == self.player_id:
				self.choose_next_available_action(bot, game)

	def choose_next_available_action(self, bot, game):
		start_priority = 1000
		if bot.location in self.priorities:
			start_priority = self.priorities[bot.location]
		ap = self.pre_actions[bot.location]
		p, action = ap.getAction(start_priority)
		self.priorities[bot.location] = p
		self.decisions[bot.location] = action
		print bot.location, "chooses", action, "::", p
		def set_move_ok():
			self.vacations[bot.location] = action[1]
			self.destinations[action[1]] = bot.location
		# now check if it's valid
		if action[0] == 'move':
			# check1: someone else moving to same spot
			if action[1] in self.destinations:
				# claim it or choose something else (recurse)
				ally_loc = self.destinations[action[1]]
				if p > self.priorities[ally_loc]:
					set_move_ok()
					# find something else for ally to do
					self.choose_next_available_action(game['robots'][ally_loc], game)
				else:
					self.choose_next_available_action(bot, game)
			# check2: ally already in this spot
			elif action[1] in game['robots'] and action[1] not in self.vacations:
				# if my movement priority higher than theirs, force movement
				if p > self.priorities[action[1]]:
					set_move_ok()
					self.choose_next_available_action(game['robots'][action[1]], game)
				# otherwise, find something else to do (recurse)
				else:
					self.choose_next_available_action(bot, game)
		elif action[0] == 'attack':
			pass
		elif action[0] == 'suicide':
			pass
		elif action[0] == 'guard':
			pass

	def setup_shared_data(self, game):
		"""Initialize shared data once per frame

			return True iff initialization happened
		"""
		if not hasattr(self, 'last_updated'):
			self.last_updated = -100
		if self.last_updated < game['turn']:
			self.last_updated = game['turn']
			# initialize anything needed at the start of the frame
			self.pre_actions = {} # map from location to ActionPriority
			self.priorities = {} # map from starting loc to value of that move
			self.destinations = {} # map from loc to previous-loc (of robot who is moving to loc)
			self.vacations = {} # map from previous loc to next-loc (which spots are being vacated)
			self.decisions = {} # final answers
			return True
		return False

