import rg
import time
import random
import itertools
import random
from collections import defaultdict
# copied default settings
settings = {
	'player_count' : 2,
	'suicide_damage': 15,
	'board_size': 19,
	'max_turns': 100,
	'default_rating': 1200,
	'user_obj_types': ('Robot',),
	'max_time_per_act': 300,
	'attack_range': (8, 10),
	'collision_damage': 5,
	'player_only_properties': ('robot_id',),
	'max_time_initialization': 2000,
	'spawn_every': 10,
	'exposed_properties': ('location', 'hp', 'player_id'),
	'spawn_per_player': 5,
	'str_limit': 50,
	'valid_commands': ('move', 'attack', 'guard', 'suicide'),
	'max_seed': 2147483647,
	'max_time_first_act': 1500,
	'robot_hp': 50,
	'spawn_coords' : [loc for loc in itertools.product(xrange(19), xrange(19)) if 'spawn' in rg.loc_types(loc)]}

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# copied fom WhiteHalmos' github
class GameState(object):
	def __init__(self, turn=0, load_state=None, next_robot_id=0, seed=None, symmetric=True):
		if seed is None:
			seed = random.randint(0, settings['max_seed'])
		self._seed = str(seed)
		self._spawn_random = random.Random(self._seed + 's')
		self._attack_random = random.Random(self._seed + 'a')
		self.symmetric = symmetric

		self.robots = {}
		self.turn = turn
		self._next_robot_id = next_robot_id

		if load_state is not None:
			self.robots = load_state.robots
			self.turn = load_state.turn

		if self.symmetric:
			self._get_spawn_locations = self._get_spawn_locations_symmetric
		else:
			self._get_spawn_locations = self._get_spawn_locations_random

	def add_robot(self, loc, player_id, hp=None, robot_id=None):
		if hp is None:
			hp = rg.settings['robot_hp']

		if robot_id is None:
			robot_id = self._next_robot_id
			self._next_robot_id += 1

		self.robots[loc] = AttrDict({
			'location': loc,
			'hp': hp,
			'player_id': player_id,
			'robot_id': robot_id
		})

	def remove_robot(self, loc):
		if self.is_robot(loc):
			del self.robots[loc]

	def is_robot(self, loc):
		return loc in self.robots

	def _get_spawn_locations_symmetric(self):
		def symmetric_loc(loc):
			return (settings['board_size'] - 1 - loc[0],
					settings['board_size'] - 1 - loc[1])
		locs1 = []
		locs2 = []
		while len(locs1) < settings['spawn_per_player']:
			loc = self._spawn_random.choice(settings['spawn_coords'])
			sloc = symmetric_loc(loc)
			if loc not in locs1 and loc not in locs2:
				if sloc not in locs1 and sloc not in locs2:
					locs1.append(loc)
					locs2.append(sloc)
		return locs1 + locs2

	def _get_spawn_locations_random(self):
		# see http://stackoverflow.com/questions/2612648/reservoir-sampling
		locations = []
		per_player = settings['spawn_per_player']
		count = per_player * settings['player_count']
		n = 0
		for loc in settings['spawn_coords']:
			n += 1
			if len(locations) < count:
				locations.append(loc)
			else:
				s = int(self._spawn_random.random() * n)
				if s < count:
					locations[s] = loc
		self._spawn_random.shuffle(locations)
		return locations

	# dest = location of robot -> its destination
	# contenders = {loc: set(locations of robots trying to move into loc)}
	def _get_contenders(self, dest):
		contenders = defaultdict(lambda: set())

		def stuck(loc):
			# Robot at loc is stuck
			# Other robots trying to move in its old locations
			# should be marked as stuck, too
			old_contenders = contenders[loc]
			contenders[loc] = set([loc])

			for contender in old_contenders:
				if contender != loc:
					stuck(contender)

		for loc in self.robots:
			contenders[dest(loc)].add(loc)

		for loc in self.robots:
			if len(contenders[dest(loc)]) > 1 or (self.is_robot(dest(loc)) and
												  dest(loc) != loc and
												  dest(dest(loc)) == loc):
				# Robot at loc is going to fail to move
				stuck(loc)

		return contenders

	# new_locations = {loc: new_loc}
	def _get_new_locations(self, dest, contenders):
		new_locations = {}

		for loc in self.robots:
			if loc != dest(loc) and loc in contenders[loc]:
				new_locations[loc] = loc
			else:
				new_locations[loc] = dest(loc)

		return new_locations

	# collisions = {loc: set(robots collided with robot at loc)}
	def _get_collisions(self, dest, contenders):
		collisions = defaultdict(lambda: set())

		for loc in self.robots:
			for loc2 in contenders[dest(loc)]:
				collisions[loc].add(loc2)
				collisions[loc2].add(loc)

		return collisions

	# damage_map = {loc: [actor_id: (actor_loc, damage)]}
	# only counts potential attack and suicide damage
	# self suicide damage is not counted
	def _get_damage_map(self, actions):
		damage_map = defaultdict(
			lambda: [{} for _ in xrange(settings['player_count'])])

		for loc, robot in self.robots.iteritems():
			actor_id = robot.player_id

			if actions[loc][0] == 'attack':
				target = actions[loc][1]
				damage = self._attack_random.randint(
					*settings['attack_range'])
				damage_map[target][actor_id][loc] = damage
			elif actions[loc][0] == 'suicide':
				damage = settings['suicide_damage']
				for target in rg.locs_around(loc):
					damage_map[target][actor_id][loc] = damage

		return damage_map

	def _apply_damage_caused(self, delta, damage_caused):
		for robot_delta in delta:
			robot_delta.damage_caused += damage_caused[robot_delta.loc]

	def _apply_spawn(self, delta):
		# clear robots on spawn
		for robot_delta in delta:
			if robot_delta.loc_end in settings['spawn_coords']:
				robot_delta.hp_end = 0

		# spawn robots
		locations = self._get_spawn_locations()
		for i in xrange(settings['spawn_per_player']):
			for player_id in xrange(settings['player_count']):
				loc = locations[player_id*settings['spawn_per_player']+i]
				delta.append(AttrDict({
					'loc': loc,
					'hp': 0,
					'player_id': player_id,
					'loc_end': loc,
					'hp_end': settings['robot_hp'],
					'damage_caused': 0
				}))

	# actions = {loc: action}
	# all actions must be valid
	# delta = [AttrDict{
	#    'loc': loc,
	#    'hp': hp,
	#    'player_id': player_id,
	#    'loc_end': loc_end,
	#    'hp_end': hp_end
	#    'damage_caused': damage_caused
	# }]
	def get_delta(self, actions, spawn=True):
		delta = []

		def dest(loc):
			if actions[loc][0] == 'move':
				return actions[loc][1]
			else:
				return loc

		contenders = self._get_contenders(dest)
		new_locations = self._get_new_locations(dest, contenders)
		collisions = self._get_collisions(dest, contenders)
		damage_map = self._get_damage_map(actions)
		damage_caused = defaultdict(lambda: 0)  # {loc: damage_caused}

		for loc, robot in self.robots.iteritems():
			robot_delta = AttrDict({
				'loc': loc,
				'hp': robot.hp,
				'player_id': robot.player_id,
				'loc_end': new_locations[loc],
				'hp_end': robot.hp,  # to be adjusted
				'damage_caused': 0  # to be adjusted
			})

			is_guard = actions[loc][0] == 'guard'

			# collision damage
			if not is_guard:
				damage = settings['collision_damage']

				for other_loc in collisions[loc]:
					if robot.player_id != self.robots[other_loc].player_id:
						robot_delta.hp_end -= damage
						damage_caused[other_loc] += damage

			# attack and suicide damage
			for player_id, player_damage_map in enumerate(
					damage_map[new_locations[loc]]):
				if player_id != robot.player_id:
					for actor_loc, damage in player_damage_map.iteritems():
						if is_guard:
							damage /= 2

						robot_delta.hp_end -= damage
						damage_caused[actor_loc] += damage

			# account for suicides
			if actions[loc][0] == 'suicide':
				robot_delta.hp_end = 0

			delta.append(robot_delta)

		self._apply_damage_caused(delta, damage_caused)

		if spawn and self.turn % settings['spawn_every'] == 0:
			self._apply_spawn(delta)

		return delta

	# delta = [AttrDict{
	#    'loc': loc,
	#    'hp': hp,
	#    'player_id': player_id,
	#    'loc_end': loc_end,
	#    'hp_end': hp_end,
	#    'damage_caused': damage_caused
	# }]
	# returns new GameState
	def apply_delta(self, delta):
		new_state = GameState(next_robot_id=self._next_robot_id,
							  turn=self.turn + 1,
							  seed=self._spawn_random.randint(
								  0, settings['max_seed']),
							  symmetric=self.symmetric)

		for delta_info in delta:
			if delta_info.hp_end > 0:
				loc = delta_info.loc

				# is this a new robot?
				if delta_info.hp > 0:
					robot_id = self.robots[loc].get('robot_id')
				else:
					robot_id = None

				new_state.add_robot(delta_info.loc_end, delta_info.player_id,
									delta_info.hp_end, robot_id)

		return new_state

	# actions = {loc: action}
	# all actions must be valid
	# returns new GameState
	def apply_actions(self, actions, spawn=True):
		delta = self.get_delta(actions, spawn)

		return self.apply_delta(delta)

	def get_scores(self):
		scores = [0 for _ in xrange(settings['player_count'])]

		for robot in self.robots.itervalues():
			scores[robot.player_id] += 1

		return scores

def valid_turns(bot):
	turn_list = [['guard']]
	for loc in rg.locs_around(bot.location, filter_out=["obstacle", "invalid"]):
		turn_list.append(['move', loc])
		turn_list.append(['attack', loc])
	return turn_list

class Robot:

	__lookahead = 4
	__act_time = 1500	# ms
	__grace_ms = 100	# stop at __act_time - __grace_ms

	def global_init(glob, game):
		if not hasattr(glob, 'global_tick'):
			glob.global_tick = -1
		if game.turn > glob.global_tick:
			glob.global_tick = game.turn
			glob.decisions = {loc: ['guard'] for loc, bot in game.robots.iteritems() if bot.player_id == glob.player_id} # final decisions mapping robot loc to an action
			glob.iter_monte(game)

	def monte_turn(glob, game_obj):
		"""executes a single turn for all robots by randomly choosing a valid action for each one
		"""
		actions = {loc: random.choice(valid_turns(bot)) for loc, bot in game_obj.robots.iteritems()}
		updated = game_obj.apply_actions(actions)
		return actions, updated

	def full_monte(glob, game_dict):
		"""executes a full monte carlo simulation either to the end of the game, or for __lookahead turns
		"""
		game_obj = GameState(load_state = game_dict)
		steps = min(Robot.__lookahead, settings['max_turns'] - game_obj.turn)
		start_actions, game_obj = glob.monte_turn(game_obj)
		for i in range(1, steps):
			_, game_obj = glob.monte_turn(game_obj)
		return start_actions, game_obj.get_scores()

	def iter_monte(glob, game_dict):
		"""keeps executing full monte carlo simulations until time runs out. chooses actions with best average results
		"""
		tstart = time.time()
		# stats maps from location to a dict which tracks actions and average scores
		stats = {
			loc: { 
				tuple(turn): {
					'score' : 0,
					'tries' : 0}
				for turn in valid_turns(bot)}
			for loc, bot in game_dict.robots.iteritems()}
		sim_count = 0
		while time.time() - tstart < (Robot.__act_time - Robot.__grace_ms) / 1000.0:
			sim_count += 1
			start_actions, scores = glob.full_monte(game_dict)
			score_swing = 2 * scores[glob.player_id] - sum(scores)
			for loc, act in start_actions.iteritems():
				if game_dict.robots[loc].player_id == glob.player_id:
					tup_act = tuple(act)
					stats[loc][tup_act]['tries'] += 1
					stats[loc][tup_act]['score'] += (score_swing - stats[loc][tup_act]['score']) / stats[loc][tup_act]['tries']
		# get best option for each bot
		for loc, bot in game_dict.robots.iteritems():
			if bot.player_id == glob.player_id:
				best_option = max(stats[loc].iteritems(), key=lambda turn : turn[1]['score'])[0]
				glob.decisions[loc] = best_option
		print "completed %d simulations" % (sim_count)


	def act(self, game):
		self.global_init(game)
		return self.decisions[self.location]
